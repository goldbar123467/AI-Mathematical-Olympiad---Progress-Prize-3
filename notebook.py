"""AIMO Progress Prize 3 - Kaggle Submission Notebook

Uses DeepSeek-R1-Distill-Qwen-32B with chain-of-thought reasoning.
Copy this into a Kaggle notebook and submit.
"""

import os
import re

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl


# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MAX_TOKENS = 8192
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
    """Extract the integer answer from model response."""
    # Look for \boxed{...} pattern first (most reliable)
    boxed_pattern = r'\\boxed\{(\d+)\}'
    matches = re.findall(boxed_pattern, response)
    if matches:
        return int(matches[-1]) % 100000

    # Fallback patterns
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

    # Last resort: find any number in the last line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'\d+', line)
        if numbers:
            return int(numbers[-1]) % 100000

    return 0


class Model:
    """DeepSeek-R1 model for math reasoning."""

    def __init__(self):
        self._llm = None
        self._tokenizer = None

    def load(self):
        """Load model using vLLM for efficient inference."""
        from vllm import LLM, SamplingParams

        print(f"Loading {MODEL_NAME}...")

        self._llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=2,  # Adjust based on available GPUs
            gpu_memory_utilization=0.95,
            max_model_len=MAX_TOKENS,
            trust_remote_code=True,
        )
        self._tokenizer = self._llm.get_tokenizer()

        self._sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=0.95,
            max_tokens=MAX_TOKENS,
            stop=["<|endoftext|>", "<|im_end|>"],
        )

        print("Model loaded successfully!")

    def predict(self, problem: str) -> int:
        """Solve a math problem and return integer answer."""
        if self._llm is None:
            self.load()

        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        outputs = self._llm.generate([prompt], self._sampling_params)
        response = outputs[0].outputs[0].text

        # Extract answer
        answer = extract_answer(response)

        return answer


model = Model()


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction for a single problem."""
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
