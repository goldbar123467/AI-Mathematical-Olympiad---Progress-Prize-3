"""
AIMO3 Competition Notebook - FAST Version
Optimized for speed: 2 samples, 1 code execution, shorter tokens
Target: <5 min per problem (50 problems in <4.5 hours)
"""

import os
import re
import io
import sys
import time
import threading
import torch
from collections import Counter
from contextlib import redirect_stdout

START_TIME = time.time()

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

# ============ SPEED-OPTIMIZED CONFIG ============
MAX_TOKENS = 1536       # Reduced from 2048
TEMPERATURE = 0.6       # Slightly lower for more focused answers
NUM_SAMPLES = 2         # Reduced from 4 - just need consensus of 2
MAX_CODE_EXECUTIONS = 1 # Reduced from 2 - one shot
CODE_TIMEOUT = 3        # Seconds per code block
TIME_LIMIT_SECONDS = 4 * 3600 + 45 * 60  # 4:45:00 (buffer)
TIME_PER_PROBLEM = 240  # Target 4 min per problem

SYSTEM_PROMPT = """You are a math olympiad expert. Solve step by step.

RULES:
1. Use Python code in ```python blocks for calculations
2. Final answer MUST be in \\boxed{N} format (integer 0-99999)
3. Be concise - focus on the solution

Example:
```python
result = pow(2, 100, 997)
print(result)
```

Solve carefully, then \\boxed{answer}."""


def execute_code_with_timeout(code: str, timeout: int = CODE_TIMEOUT) -> str:
    """Thread-based timeout for Kaggle compatibility."""
    result = {"output": None, "error": None}

    def run():
        namespace = {
            '__builtins__': __builtins__,
            'math': __import__('math'),
            'itertools': __import__('itertools'),
            'functools': __import__('functools'),
            'fractions': __import__('fractions'),
        }
        try:
            namespace['sympy'] = __import__('sympy')
        except:
            pass

        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, namespace)
            result["output"] = out.getvalue().strip()
            if not result["output"]:
                for v in ['result', 'answer', 'ans']:
                    if v in namespace and namespace[v] is not None:
                        result["output"] = str(namespace[v])
                        break
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()
    t.join(timeout)

    if t.is_alive():
        return "Timeout"
    if result["error"]:
        return f"Error: {result['error']}"
    return result["output"] or "(no output)"


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r'```python\n(.*?)```', text, re.DOTALL)


def extract_answer(response: str) -> int:
    # \boxed{N}
    matches = re.findall(r'\\boxed\{(\d+)\}', response)
    if matches:
        return int(matches[-1]) % 100000

    # Fallback patterns
    for pattern in [r'boxed\{(\d+)\}', r'[Aa]nswer[:\s]+(\d+)', r'= (\d+)\s*$']:
        m = re.search(pattern, response)
        if m:
            return int(m.group(1)) % 100000

    # Last number in response
    nums = re.findall(r'\b(\d{1,5})\b', response)
    if nums:
        return int(nums[-1]) % 100000
    return 0


MODEL_PATHS = [
    "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2",
    "/kaggle/input/deepseek-r1/deepseek-r1-distill-qwen-7b",
]


def find_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p) and os.path.isfile(os.path.join(p, "config.json")):
            return p
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files and "deepseek" in root.lower():
            return root
        if root.count(os.sep) > 6:
            break
    raise FileNotFoundError("Model not found")


class Model:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._count = 0

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = find_model_path()
        print(f"Loading {path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
        print("Model loaded!")

    def _generate(self, messages, max_tokens=MAX_TOKENS):
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=TEMPERATURE,
                do_sample=True, top_p=0.9, pad_token_id=self._tokenizer.eos_token_id
            )
        return self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _solve(self, problem: str, time_limit: float) -> int:
        start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        response = self._generate(messages)

        # One round of code execution if needed
        if time.time() - start < time_limit:
            codes = extract_code_blocks(response)
            if codes and not re.search(r'\\boxed\{\d+\}', response):
                results = [execute_code_with_timeout(c) for c in codes[:2]]
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Results: {results}\nGive final \\boxed{{answer}}"})
                response += "\n" + self._generate(messages, max_tokens=512)

        return extract_answer(response)

    def predict(self, problem: str) -> int:
        if self._model is None:
            self.load()

        self._count += 1
        start = time.time()

        # Time management
        elapsed = time.time() - START_TIME
        remaining = TIME_LIMIT_SECONDS - elapsed

        if remaining < 180:
            n_samples = 1
            print(f"URGENT: {remaining:.0f}s left")
        else:
            n_samples = NUM_SAMPLES

        time_per_sample = min(TIME_PER_PROBLEM / n_samples, remaining / 2)

        answers = []
        for i in range(n_samples):
            try:
                ans = self._solve(problem, time_per_sample)
                answers.append(ans)
                # Early exit on agreement
                if len(answers) >= 2 and answers[0] == answers[1]:
                    break
            except Exception as e:
                print(f"Sample {i+1} error: {e}")

        if not answers:
            return 0

        # Majority vote
        counter = Counter(answers)
        best, count = counter.most_common(1)[0]
        conf = count / len(answers) * 100

        print(f"Problem {self._count}: answer={best} ({conf:.0f}% conf) in {time.time()-start:.1f}s")
        return best


model = Model()


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    problem_text = problem.item(0)
    try:
        prediction = model.predict(problem_text)
    except Exception as e:
        print(f"Error on {id_}: {e}")
        prediction = 0
    return pl.DataFrame({'id': id_, 'answer': prediction})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',))
