"""
AIMO3 Competition Notebook - Fixed Version
Key fixes:
1. Thread-based timeout instead of signal (works in Kaggle subprocess)
2. Optimized time management for 50 problems in 5 hours
3. Reduced sampling for speed
"""

import os
import re
import io
import sys
import time
import threading
import multiprocessing
import torch
from collections import Counter
from contextlib import redirect_stdout


START_TIME = time.time()

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

# ============ CONFIGURATION ============
MAX_TOKENS = 2048  # Reduced for speed
TEMPERATURE = 0.7
NUM_SAMPLES = 4  # Reduced from 8 for speed
MAX_CODE_EXECUTIONS = 2  # Reduced from 3
CODE_TIMEOUT = 3  # Seconds per code block
TIME_LIMIT_SECONDS = 4 * 3600 + 50 * 60  # 4:50:00 (with 9 min buffer)
TIME_PER_PROBLEM = 300  # Target 5 min per problem (for 50 problems)

SYSTEM_PROMPT = """You are a math olympiad expert. Solve the problem step by step.

IMPORTANT RULES:
1. For ANY calculation, write Python code in ```python blocks
2. Code will be executed and results returned to you
3. Your final answer MUST be an integer in \\boxed{N} format
4. Answer must be between 0 and 99999

Example code usage:
```python
result = pow(2, 100, 997)  # Modular exponentiation
print(result)
```

Solve carefully, verify with code, then give \\boxed{answer}."""


def execute_code_with_timeout(code: str, timeout: int = CODE_TIMEOUT) -> str:
    """Execute Python code with thread-based timeout (works in Kaggle)."""
    result_container = {"output": None, "error": None}

    def run_code():
        namespace = {
            '__builtins__': __builtins__,
            'math': __import__('math'),
            'cmath': __import__('cmath'),
            'itertools': __import__('itertools'),
            'functools': __import__('functools'),
            'fractions': __import__('fractions'),
            'decimal': __import__('decimal'),
            'collections': __import__('collections'),
            'random': __import__('random'),
        }
        try:
            namespace['sympy'] = __import__('sympy')
        except ImportError:
            pass

        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exec(code, namespace)
            result = output.getvalue().strip()

            if not result:
                for var in ['result', 'answer', 'ans', 'res']:
                    if var in namespace and namespace[var] is not None:
                        result = str(namespace[var])
                        break

            result_container["output"] = result if result else "(no output)"
        except Exception as e:
            result_container["error"] = f"{type(e).__name__}: {str(e)}"

    thread = threading.Thread(target=run_code)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return "Error: Code timed out"

    if result_container["error"]:
        return f"Error: {result_container['error']}"

    return result_container["output"] or "(no output)"


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from model output."""
    pattern = r'```python\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def extract_answer(response: str) -> int:
    """Extract final numerical answer from response."""
    # Try \boxed{N} first (most reliable)
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return int(matches[-1]) % 100000

    # Try boxed with text: \boxed{42}
    boxed_pattern2 = r'boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern2, response)
    if matches:
        return int(matches[-1]) % 100000

    # Fallback patterns
    patterns = [
        r'[Ff]inal [Aa]nswer[:\s]+(\d+)',
        r'[Aa]nswer[:\s]+(\d+)',
        r'[Aa]nswer is[:\s]+(\d+)',
        r'\*\*(\d+)\*\*\s*$',
        r'= (\d+)\s*$'
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return int(match.group(1)) % 100000

    # Last resort: find last reasonable number in last few lines
    lines = response.strip().split('\n')[-10:]
    for line in reversed(lines):
        # Look for standalone numbers that could be answers
        numbers = re.findall(r'\b(\d{1,5})\b', line)
        if numbers:
            num = int(numbers[-1])
            if 0 <= num <= 99999:
                return num

    return 0


MODEL_PATHS = [
    "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2",
    "/kaggle/input/deepseek-r1/deepseek-r1-distill-qwen-7b",
    "/kaggle/input/deepseek-r1",
]


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            if os.path.isfile(os.path.join(path, "config.json")):
                print(f"Using model: {path}")
                return path

    # Search for it
    print("Searching for model...")
    for root, dirs, files in os.walk("/kaggle/input"):
        if "config.json" in files and "deepseek" in root.lower():
            print(f"Found model at: {root}")
            return root
        if root.count(os.sep) > 6:
            break

    raise FileNotFoundError("DeepSeek model not found")


class Model:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._problem_count = 0
        self._problem_start_time = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = find_model_path()
        print(f"Loading {model_path}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print("Model loaded!")

    def _get_time_budget(self) -> tuple[float, int]:
        """Calculate remaining time and appropriate number of samples."""
        elapsed = time.time() - START_TIME
        remaining = TIME_LIMIT_SECONDS - elapsed

        # Estimate problems remaining (assume 50 total)
        estimated_remaining_problems = max(1, 50 - self._problem_count)
        time_per_problem = remaining / estimated_remaining_problems

        # Adaptive sampling
        if remaining < 120:  # Less than 2 min left
            num_samples = 1
            print(f"CRITICAL: {remaining:.0f}s left! Single sample mode")
        elif remaining < 300:  # Less than 5 min left
            num_samples = 1
            print(f"URGENT: {remaining:.0f}s left, single sample")
        elif time_per_problem < 120:  # Less than 2 min per problem
            num_samples = 2
            print(f"Time pressure: {time_per_problem:.0f}s/problem, 2 samples")
        elif time_per_problem < 240:  # Less than 4 min per problem
            num_samples = 3
        else:
            num_samples = NUM_SAMPLES

        return remaining, num_samples

    def _generate_once(self, messages: list[dict], max_tokens: int = None) -> str:
        """Generate a single response from messages."""
        if max_tokens is None:
            max_tokens = MAX_TOKENS

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response

    def _solve_with_code(self, problem: str, time_limit: float) -> tuple[str, int]:
        """Solve problem with code execution support."""
        solve_start = time.time()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        full_response = ""

        for iteration in range(MAX_CODE_EXECUTIONS + 1):
            # Check time limit for this problem
            if time.time() - solve_start > time_limit:
                break

            response = self._generate_once(messages)
            full_response += response

            # Check for code blocks
            code_blocks = extract_code_blocks(response)

            # If we have an answer and no new code, we're done
            has_answer = bool(re.search(r'\\boxed\{\d+\}', response))
            if has_answer or not code_blocks:
                break

            # Execute code blocks (limit to first 2 for speed)
            results = []
            for i, code in enumerate(code_blocks[:2]):
                result = execute_code_with_timeout(code, CODE_TIMEOUT)
                results.append(f"Output {i+1}: {result}")

            execution_output = "\n".join(results)

            # Continue conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Code results:\n{execution_output}\n\nContinue and give final answer in \\boxed{{N}} format."
            })
            full_response += f"\n[EXECUTED]\n{execution_output}\n"

        answer = extract_answer(full_response)
        return full_response, answer

    def predict(self, problem: str) -> int:
        if self._model is None:
            self.load()

        self._problem_count += 1
        self._problem_start_time = time.time()

        remaining, num_samples = self._get_time_budget()

        # Calculate time limit for this problem
        time_limit_per_sample = min(TIME_PER_PROBLEM / num_samples, remaining / (num_samples + 1))

        answers = []

        for sample_idx in range(num_samples):
            try:
                _, answer = self._solve_with_code(problem, time_limit_per_sample)
                answers.append(answer)

                # Early exit if we have consistent answers
                if len(answers) >= 2:
                    counter = Counter(answers)
                    top_answer, top_count = counter.most_common(1)[0]
                    if top_count >= 2 and top_count / len(answers) >= 0.5:
                        # Good enough consensus
                        break

            except Exception as e:
                print(f"Sample {sample_idx+1} error: {e}")
                continue

        if not answers:
            print(f"Problem {self._problem_count}: No valid answers!")
            return 0

        # Majority vote
        counter = Counter(answers)
        top_answer, top_count = counter.most_common(1)[0]
        confidence = top_count / len(answers)

        problem_time = time.time() - self._problem_start_time
        print(f"Problem {self._problem_count}: answer={top_answer} ({confidence:.0%} conf) in {problem_time:.1f}s")

        return top_answer


model = Model()


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Main prediction function called by Kaggle evaluation server."""
    id_ = id_.item(0)
    problem_text: str = problem.item(0)

    try:
        prediction = model.predict(problem_text)
    except Exception as e:
        print(f"Error on {id_}: {e}")
        prediction = 0

    return pl.DataFrame({'id': id_, 'answer': prediction})


# Entry point
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',))
