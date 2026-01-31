"""Base solver interface for AIMO problems."""

from abc import ABC, abstractmethod
import re


class BaseSolver(ABC):
    """Abstract base class for math problem solvers."""

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def solve(self, problem: str) -> int:
        """Solve a math problem and return integer answer."""
        pass

    def extract_answer(self, response: str) -> int:
        """Extract integer answer from model response."""
        patterns = [
            r'\\boxed\{(\d+)\}',
            r'\*\*(\d+)\*\*',
            r'[Aa]nswer[:\s]+(\d+)',
            r'[Rr]esult[:\s]+(\d+)',
            r'[Ff]inal[:\s]+(\d+)',
            r'= (\d+)\s*$',
            r'(\d+)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                try:
                    answer = int(match.group(1))
                    return answer % 100000  # 5-digit constraint
                except ValueError:
                    continue

        return 0  # Fallback

    def classify_problem(self, problem: str) -> str:
        """Classify problem into category."""
        problem_lower = problem.lower()

        keywords = {
            'geometry': ['triangle', 'circle', 'angle', 'point', 'line',
                        'perpendicular', 'circumcircle', 'incircle'],
            'number_theory': ['divisor', 'prime', 'modulo', 'integer',
                             'remainder', 'divides', 'gcd', 'lcm'],
            'combinatorics': ['count', 'permutation', 'sequence', 'ways',
                             'arrangement', 'subset', 'tournament'],
            'algebra': ['polynomial', 'equation', 'root', 'function',
                       'sum', 'product', 'variable']
        }

        scores = {cat: 0 for cat in keywords}
        for cat, words in keywords.items():
            for word in words:
                if word in problem_lower:
                    scores[cat] += 1

        if max(scores.values()) == 0:
            return 'general'

        return max(scores, key=scores.get)
