"""Ensemble solver combining multiple solving strategies."""

from typing import List, Dict, Any
from collections import Counter

from .base import BaseSolver


class EnsembleSolver(BaseSolver):
    """Ensemble of multiple solvers with voting."""

    def __init__(self, solvers: List[BaseSolver], weights: List[float] = None):
        super().__init__()
        self.solvers = solvers
        self.weights = weights or [1.0] * len(solvers)

    def solve(self, problem: str) -> int:
        """Solve using weighted voting across all solvers."""
        answers_with_weights = []

        for solver, weight in zip(self.solvers, self.weights):
            try:
                answer = solver.solve(problem)
                answers_with_weights.append((answer, weight))
            except Exception as e:
                print(f"Solver {solver.__class__.__name__} failed: {e}")
                continue

        if not answers_with_weights:
            return 0

        # Weighted voting
        vote_counts: Dict[int, float] = {}
        for answer, weight in answers_with_weights:
            vote_counts[answer] = vote_counts.get(answer, 0) + weight

        # Return answer with highest weighted votes
        return max(vote_counts, key=vote_counts.get)


class CategoryAwareSolver(BaseSolver):
    """Solver that routes to specialized solvers based on problem category."""

    def __init__(self, default_solver: BaseSolver,
                 category_solvers: Dict[str, BaseSolver] = None):
        super().__init__()
        self.default_solver = default_solver
        self.category_solvers = category_solvers or {}

    def solve(self, problem: str) -> int:
        """Route to appropriate solver based on problem category."""
        category = self.classify_problem(problem)

        if category in self.category_solvers:
            solver = self.category_solvers[category]
        else:
            solver = self.default_solver

        return solver.solve(problem)
