"""AIMO Progress Prize 3 - QwQ-32B Version

Alternative using Qwen's QwQ-32B-Preview which has strong math reasoning.
"""

import os
import re

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl


MODEL_NAME = "Qwen/QwQ-32B-Preview"
MAX_TOKENS = 16384  # QwQ supports longer context
TEMPERATURE = 0.0  # Greedy for consistency


SYSTEM_PROMPT = """You are a mathematical reasoning expert. Solve this olympiad problem step by step.

Rules:
- Think through each step carefully
- Show all work and reasoning
- Verify your answer before submitting
- Put your final integer answer in \\boxed{}"""


def extract_answer(response: str) -> int:
    """Extract integer answer from response."""
    # \boxed{} is standard
    boxed = re.findall(r'\\boxed\{(\d+)\}', response)
    if boxed:
        return int(boxed[-1]) % 100000

    # Fallbacks
    for pattern in [r'answer[:\s]+(\d+)', r'\*\*(\d+)\*\*', r'= (\d+)']:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1)) % 100000

    # Last number in response
    nums = re.findall(r'\d+', response)
    if nums:
        return int(nums[-1]) % 100000

    return 0


class Model:
    def __init__(self):
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

    def load(self):
        from vllm import LLM, SamplingParams

        print(f"Loading {MODEL_NAME}...")

        self._llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.95,
            max_model_len=MAX_TOKENS,
            trust_remote_code=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
        print("Model loaded!")

    def predict(self, problem: str) -> int:
        if self._llm is None:
            self.load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self._llm.generate([prompt], self._sampling_params)
        response = outputs[0].outputs[0].text

        return extract_answer(response)


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
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )
