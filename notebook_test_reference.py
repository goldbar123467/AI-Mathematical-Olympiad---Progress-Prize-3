"""
AIMO3 - Test on Reference Problems
Run this to validate your solution on the 10 known reference problems before submitting.
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

# ============ SPEED-OPTIMIZED CONFIG ============
MAX_TOKENS = 1536
TEMPERATURE = 0.6
NUM_SAMPLES = 2
MAX_CODE_EXECUTIONS = 1
CODE_TIMEOUT = 3

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
    matches = re.findall(r'\\boxed\{(\d+)\}', response)
    if matches:
        return int(matches[-1]) % 100000
    for pattern in [r'boxed\{(\d+)\}', r'[Aa]nswer[:\s]+(\d+)', r'= (\d+)\s*$']:
        m = re.search(pattern, response)
        if m:
            return int(m.group(1)) % 100000
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

    def _solve(self, problem: str, time_limit: float = 300) -> int:
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

        answers = []
        for i in range(NUM_SAMPLES):
            try:
                ans = self._solve(problem)
                answers.append(ans)
                if len(answers) >= 2 and answers[0] == answers[1]:
                    break
            except Exception as e:
                print(f"Sample {i+1} error: {e}")

        if not answers:
            return 0

        counter = Counter(answers)
        best, count = counter.most_common(1)[0]
        conf = count / len(answers) * 100

        print(f"Problem {self._count}: answer={best} ({conf:.0f}% conf) in {time.time()-start:.1f}s")
        return best


# ============ REFERENCE TEST MODE ============
if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("AIMO3 - Testing on 10 Reference Problems")
    print("=" * 60)

    # Load reference problems and answers
    ref = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv')

    model = Model()
    results = []

    for idx, row in ref.iterrows():
        problem_id = row['id']
        problem = row['problem']
        correct_answer = row['answer']

        print(f"\n--- Problem {idx+1}/10: {problem_id} ---")
        print(f"Expected: {correct_answer}")

        predicted = model.predict(problem)
        is_correct = predicted == correct_answer

        results.append({
            'id': problem_id,
            'predicted': predicted,
            'correct': correct_answer,
            'match': is_correct
        })

        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"Predicted: {predicted} {status}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    correct_count = sum(1 for r in results if r['match'])
    print(f"\n{'ID':10} | {'Predicted':>10} | {'Correct':>10} | Status")
    print("-" * 50)
    for r in results:
        status = "✓" if r['match'] else "✗"
        print(f"{r['id']:10} | {r['predicted']:>10} | {r['correct']:>10} | {status}")

    print("-" * 50)
    print(f"Score: {correct_count}/10 ({correct_count*10}%)")
    print(f"Total time: {time.time() - START_TIME:.1f}s")
