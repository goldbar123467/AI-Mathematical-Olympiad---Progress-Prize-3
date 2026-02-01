import os
import re
import io
import sys
import time
import torch
from collections import Counter
from contextlib import redirect_stdout

START_TIME = time.time()

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

MAX_TOKENS = 3072
TEMPERATURE = 0.7
NUM_SAMPLES = 8
MAX_CODE_EXECUTIONS = 3  # Max code execution rounds per sample
TIME_LIMIT_SECONDS = 4 * 3600 + 59 * 60  # 4:59:00 competition limit

SYSTEM_PROMPT = """You are a world-class mathematician solving International Mathematical Olympiad problems.

## CRITICAL: USE PYTHON FOR CALCULATIONS
When you need to compute anything non-trivial (modular arithmetic, large numbers, combinatorics, checking cases), write Python code in ```python blocks. The code will be executed and results returned to you.

Example:
```python
# Calculate 2^100 mod 997
result = pow(2, 100, 997)
print(result)
```

DO NOT guess numerical results. ALWAYS compute them with code.

## PROBLEM-SOLVING FRAMEWORK

### Step 1: UNDERSTAND
- What quantities are given?
- What is being asked?
- What type of problem is this? (geometry, number theory, combinatorics, algebra)

### Step 2: EXPLORE WITH CODE
- Write Python to try small cases
- Use code to find patterns
- Verify conjectures computationally

### Step 3: SOLVE
- Execute your approach step-by-step
- Use Python for ALL calculations
- State theorems you apply

### Step 4: VERIFY WITH CODE
- Write code to check your answer
- Test edge cases
- Confirm answer is in range [0, 99999]

### Step 5: ANSWER
- State your final answer
- Put it in \\boxed{N} format

## MATH CODE PATTERNS

**Modular arithmetic:**
```python
pow(base, exp, mod)  # Fast modular exponentiation
```

**Combinatorics:**
```python
from math import comb, factorial
comb(n, k)  # n choose k
```

**Number theory:**
```python
from math import gcd
from functools import reduce
def lcm(a, b): return a * b // gcd(a, b)
```

**Brute force search:**
```python
for n in range(1, 1000):
    if condition(n):
        print(n)
        break
```

## CRITICAL
Your final answer MUST be a single integer between 0 and 99999.
Express it as: \\boxed{YOUR_ANSWER}"""


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from model output."""
    pattern = r'```python\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def execute_code(code: str, timeout: float = 5.0) -> str:
    """Execute Python code and return output."""
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
        'sympy': None,  # Will try to import
    }

    # Try to import sympy if available
    try:
        namespace['sympy'] = __import__('sympy')
    except ImportError:
        pass

    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exec(code, namespace)
        result = output.getvalue().strip()

        # Check for result/answer variables if no print output
        if not result:
            for var in ['result', 'answer', 'ans', 'res']:
                if var in namespace and namespace[var] is not None:
                    result = str(namespace[var])
                    break

        return result if result else "(code executed, no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


def extract_answer(response: str) -> int:
    """Extract final numerical answer from response."""
    # Try \boxed{N} first
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return int(matches[-1]) % 100000

    # Fallback patterns
    patterns = [
        r'[Aa]nswer[:\s]+(\d+)',
        r'[Ff]inal[:\s]+(\d+)',
        r'\*\*(\d+)\*\*',
        r'= (\d+)\s*$'
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return int(match.group(1)) % 100000

    # Last resort: find last number
    lines = response.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'\d+', line)
        if numbers:
            return int(numbers[-1]) % 100000
    return 0


MODEL_PATH = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2"


def find_model_path():
    if os.path.exists(MODEL_PATH):
        print(f"Using model: {MODEL_PATH}")
        return MODEL_PATH
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")


class Model:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = find_model_path()
        print(f"Loading {model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
        print("Model loaded!")

    def _generate_once(self, messages: list[dict]) -> str:
        """Generate a single response from messages."""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response

    def _solve_with_tir(self, problem: str) -> tuple[str, int]:
        """Solve problem with Tool-Integrated Reasoning (code execution)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        full_response = ""

        for iteration in range(MAX_CODE_EXECUTIONS + 1):
            response = self._generate_once(messages)
            full_response += response

            # Check for code blocks
            code_blocks = extract_code_blocks(response)

            if not code_blocks:
                # No code to execute, we're done
                break

            # Execute all code blocks
            results = []
            for i, code in enumerate(code_blocks):
                result = execute_code(code)
                results.append(f"Code block {i+1} output:\n{result}")

            execution_output = "\n\n".join(results)

            # Add to conversation and continue
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Execution results:\n{execution_output}\n\nContinue solving using these results. When done, put your final answer in \\boxed{{N}} format."
            })
            full_response += f"\n\n[CODE EXECUTED]\n{execution_output}\n\n"

        answer = extract_answer(full_response)
        return full_response, answer

    def predict(self, problem: str) -> int:
        if self._model is None:
            self.load()

        elapsed = time.time() - START_TIME
        remaining = TIME_LIMIT_SECONDS - elapsed

        # Adaptive sampling based on time remaining
        if remaining < 300:  # Less than 5 min left
            num_samples = 1
            print(f"URGENT: {remaining:.0f}s left, single sample mode")
        elif remaining < 900:  # Less than 15 min left
            num_samples = 4
            print(f"WARNING: {remaining:.0f}s left, reduced to 4 samples")
        else:
            num_samples = NUM_SAMPLES

        answers = []

        for sample_idx in range(num_samples):
            try:
                _, answer = self._solve_with_tir(problem)
                answers.append(answer)
            except Exception as e:
                print(f"Sample {sample_idx} error: {e}")
                answers.append(0)

        # Majority vote with confidence tracking
        counter = Counter(answers)
        top_answer, top_count = counter.most_common(1)[0]
        confidence = top_count / len(answers)

        if confidence < 0.5:
            print(f"Low confidence ({confidence:.0%}): {dict(counter)}")
        else:
            print(f"Confident ({confidence:.0%}): answer={top_answer}")

        return top_answer


model = Model()


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    problem_text: str = problem.item(0)
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
