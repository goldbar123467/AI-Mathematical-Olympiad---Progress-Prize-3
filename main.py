"""AIMO Progress Prize 3 - Main Entry Point

This is the Kaggle submission entry point that implements the inference server.
"""

import os
import sys
import polars as pl

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_evaluation.core.templates import InferenceServer
from src.solvers.base import BaseSolver
from src.solvers.cot_solver import ChainOfThoughtSolver, SelfConsistencySolver


class AIMO3Solver(InferenceServer):
    """Main solver for AIMO Progress Prize 3."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.solver = None

    def load(self):
        """Load model and initialize solver."""
        # Use vLLM for H100 inference
        from src.models.vllm_model import VLLMModel

        print("Loading model...")
        self.model = VLLMModel(
            model_name=os.environ.get(
                "MODEL_NAME",
                "Qwen/Qwen2.5-Math-72B-Instruct"
            ),
            tensor_parallel_size=int(os.environ.get("TP_SIZE", "4")),
            gpu_memory_utilization=0.95,
            max_model_len=8192
        )
        self.model.load()

        # Use self-consistency for robustness
        self.solver = SelfConsistencySolver(
            model=self.model,
            n_samples=int(os.environ.get("N_SAMPLES", "5")),
            temperature=0.7
        )
        print("Model loaded successfully")

    def predict(self, id: str, problem: str) -> dict:
        """Solve a single problem."""
        if self.solver is None:
            self.load()

        try:
            answer = self.solver.solve(problem)
        except Exception as e:
            print(f"Error solving problem {id}: {e}")
            answer = 0

        # Ensure 5-digit constraint
        answer = int(answer) % 100000

        return {"id": id, "answer": answer}


def evaluate_local(test_path: str, output_path: str = None):
    """Run local evaluation on test set."""
    solver = AIMO3Solver()
    solver.load()

    test_df = pl.read_csv(test_path)
    results = []

    for row in test_df.iter_rows(named=True):
        problem_id = row["id"]
        problem = row["problem"]

        print(f"Solving problem {problem_id}...")
        result = solver.predict(problem_id, problem)
        results.append(result)

        # Check against answer if available
        if "answer" in row:
            correct = result["answer"] == row["answer"]
            print(f"  Answer: {result['answer']} | Expected: {row['answer']} | {'OK' if correct else 'WRONG'}")
        else:
            print(f"  Answer: {result['answer']}")

    # Save results
    if output_path:
        results_df = pl.DataFrame(results)
        results_df.write_csv(output_path)
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIMO 3 Solver")
    parser.add_argument("--test", type=str, help="Path to test CSV for local evaluation")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output path")
    parser.add_argument("--server", action="store_true", help="Run as inference server")

    args = parser.parse_args()

    if args.test:
        # Local evaluation mode
        evaluate_local(args.test, args.output)
    elif args.server or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # Kaggle submission mode
        server = AIMO3Solver()
        server.run()
    else:
        print("Usage:")
        print("  Local eval:  python main.py --test data/reference.csv")
        print("  Server mode: python main.py --server")
