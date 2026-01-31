"""AIMO Progress Prize 3 - Offline Submission

Uses pre-downloaded model from Kaggle Models (no internet required).
Add the model to your notebook inputs first!
"""

import os
import re
import torch

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl


# Use Kaggle Model path - ADD THE MODEL TO YOUR NOTEBOOK INPUTS FIRST
# Search for "deepseek" in Kaggle Models and add it
MODEL_NAME = "/kaggle/input/deepseek-r1-distill-qwen-7b/transformers/default/1"

# Fallback paths to try
MODEL_PATHS = [
    "/kaggle/input/deepseek-r1-distill-qwen-7b/transformers/default/1",
    "/kaggle/input/deepseek-r1-distill-qwen-7b",
    "/kaggle/input/deepseek-ai/deepseek-r1-distill-qwen-7b/transformers/default/1",
]

MAX_TOKENS = 4096
TEMPERATURE = 0.1


SYSTEM_PROMPT = """You are an expert mathematician solving International Mathematical Olympiad problems.

Instructions:
1. Read the problem carefully
2. Think step by step, showing all reasoning
3. Verify each step before proceeding
4. Double-check your final answer
5. Express your final answer as a single integer inside \\boxed{}

Important: The answer must be an integer between 0 and 99999."""


def extract_answer(response: str) -> int:
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return int(matches[-1]) % 100000

    patterns = [
        r'[Aa]nswer[:\s]+(\d+)',
        r'[Rr]esult[:\s]+(\d+)',
        r'[Ff]inal[:\s]+(\d+)',
        r'\*\*(\d+)\*\*',
        r'= (\d+)\s*$',
    ]

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


def find_model_path():
    """Find the model path from available inputs."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path

    # List what's available
    print("Available inputs:")
    input_dir = "/kaggle/input"
    if os.path.exists(input_dir):
        for item in os.listdir(input_dir):
            print(f"  - {item}")

    raise FileNotFoundError("Model not found! Add DeepSeek-R1-Distill-Qwen-7B to notebook inputs.")


class Model:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = find_model_path()
        print(f"Loading model from {model_path}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        print("Model loaded successfully!")

    def predict(self, problem: str) -> int:
        if self._model is None:
            self.load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return extract_answer(response)


model = Model()


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    problem_text: str = problem.item(0)

    try:
        prediction = model.predict(problem_text)
    except Exception as e:
        print(f"Error on problem {id_}: {e}")
        prediction = 0

    return pl.DataFrame({'id': id_, 'answer': prediction})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )
