import os
import re
import torch
from collections import Counter

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl

MAX_TOKENS = 4096
TEMPERATURE = 0.7
NUM_SAMPLES = 16  # Generate multiple reasoning traces, majority vote

SYSTEM_PROMPT = """You are a world-class mathematician solving International Mathematical Olympiad problems.

## PROBLEM-SOLVING FRAMEWORK

### Step 1: UNDERSTAND
- What quantities are given?
- What is being asked?
- What type of problem is this? (geometry, number theory, combinatorics, algebra, functional equations)

### Step 2: EXPLORE
- Try small cases or specific values
- Look for patterns
- Consider what techniques apply

### Step 3: SOLVE
- Execute your approach step-by-step
- Show all calculations clearly
- State any theorems or lemmas you use

### Step 4: VERIFY
- Check with a different method or edge case
- Verify the answer makes sense
- Confirm it's an integer in range [0, 99999]

### Step 5: ANSWER
- State your final answer clearly
- Put it in \\boxed{N} format

## MATH-SPECIFIC TIPS

**Number Theory:**
- For "remainder when divided by N": work in mod N throughout
- Factor large numbers, use CRT for composite moduli
- Check divisibility patterns

**Geometry:**
- Set up coordinates if synthetic approach is unclear
- Use trigonometry for angles
- Apply power of a point, radical axes

**Combinatorics:**
- Verify formula with small cases (n=1,2,3)
- Use generating functions for complex counting
- Check for overcounting

**Algebra/Functions:**
- Substitute special values: f(0), f(1), f(-1)
- Look for functional equation patterns
- Check if function is linear, multiplicative, etc.

## CRITICAL
Your final answer MUST be a single integer between 0 and 99999.
Express it as: \\boxed{YOUR_ANSWER}"""


def extract_answer(response: str) -> int:
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return int(matches[-1]) % 100000
    patterns = [r'[Aa]nswer[:\s]+(\d+)', r'[Ff]inal[:\s]+(\d+)', r'\*\*(\d+)\*\*', r'= (\d+)\s*$']
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return int(match.group(1)) % 100000
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
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True)
        print("Model loaded!")

    def predict(self, problem: str) -> int:
        if self._model is None:
            self.load()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        answers = []
        with torch.no_grad():
            for _ in range(NUM_SAMPLES):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.eos_token_id
                )
                response = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                ans = extract_answer(response)
                answers.append(ans)

        # Majority vote
        vote = Counter(answers).most_common(1)[0][0]
        return vote


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
