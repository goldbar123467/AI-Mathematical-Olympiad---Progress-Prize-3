"""Chain-of-Thought solver for AIMO problems."""

from .base import BaseSolver


SYSTEM_PROMPT = """You are an expert mathematician competing in the International Mathematical Olympiad.

Instructions:
1. Read the problem carefully and identify the key mathematical concepts
2. Think step by step, showing all intermediate calculations
3. Verify your reasoning at each step
4. Double-check your final answer
5. Express your final answer as a single integer

Format your final answer as: \\boxed{answer}
"""


class ChainOfThoughtSolver(BaseSolver):
    """Solver using chain-of-thought prompting."""

    def __init__(self, model, temperature: float = 0.1, max_tokens: int = 8192):
        super().__init__(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, problem: str) -> int:
        """Solve problem using chain-of-thought reasoning."""
        if self.model is None:
            raise ValueError("Model not initialized")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        response = self.model.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return self.extract_answer(response)

    def solve_with_retry(self, problem: str, max_retries: int = 3) -> int:
        """Solve with multiple attempts at different temperatures."""
        temperatures = [0.1, 0.3, 0.5]

        for i, temp in enumerate(temperatures[:max_retries]):
            self.temperature = temp
            answer = self.solve(problem)
            if answer != 0:  # Found a valid answer
                return answer

        return 0


class SelfConsistencySolver(BaseSolver):
    """Solver using self-consistency (majority voting)."""

    def __init__(self, model, n_samples: int = 8, temperature: float = 0.7):
        super().__init__(model)
        self.n_samples = n_samples
        self.temperature = temperature
        self.cot_solver = ChainOfThoughtSolver(model, temperature=temperature)

    def solve(self, problem: str) -> int:
        """Solve using majority voting over multiple samples."""
        answers = []

        for _ in range(self.n_samples):
            answer = self.cot_solver.solve(problem)
            answers.append(answer)

        # Majority vote
        from collections import Counter
        counter = Counter(answers)
        most_common = counter.most_common(1)

        if most_common:
            return most_common[0][0]
        return 0
