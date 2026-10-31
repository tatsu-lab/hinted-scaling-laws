"""AIME benchmark loader."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hinted.benchmarks import Benchmark, BenchmarkProblem, EvaluationResult
from hinted.benchmarks.aime_grader import grade_aime_response


@dataclass
class AIMEProblem(BenchmarkProblem):
    """An AIME problem with AIME-specific prompt templates."""

    def question_prompt(self) -> str:
        return (
            "Put your final answer within \\boxed{}.\n"
            "The answer is an integer between 0 and 999 inclusive.\n\n"
            f"Problem:\n{self.question.strip()}"
        )

    def hinted_prompt_from_text(self, hint_text: str) -> str:
        return (
            "You will be given a problem and a hint to the problem.\n"
            "Put your final answer within \\boxed{}.\n"
            "The answer is an integer between 0 and 999 inclusive.\n\n"
            f"Problem:\n{self.question.strip()}\n\n"
            f"Hint:\n{hint_text.strip()}"
        )


class AIMEBenchmark(Benchmark):
    """
    AIME benchmark 2025/26.
    """

    def __init__(self, data_path: str = "data/benchmarks/aime2025_2026.jsonl") -> None:
        """Initialize AIME benchmark.

        Args:
            data_path: Path to the problems JSONL file
        """
        super().__init__(name="aime2025_2026")
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> List[BenchmarkProblem]:
        """Load AIME problems from disk.

        Args:
            limit: Optional limit on number of problems to load

        Returns:
            List of benchmark problems
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"AIME data not found at {self.data_path}. "
                "Format: JSONL rows with problem_id, question, answer, source."
            )

        with open(self.data_path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if limit is not None:
            rows = rows[:limit]

        return [
            AIMEProblem(
                problem_id=row["problem_id"],
                question=row["question"],
                answer=str(row["answer"]),
                metadata={"source": row["source"], "problem_number": idx + 1},
            )
            for idx, row in enumerate(rows)
        ]

    def verify(self, prediction: str, ground_truth: str) -> EvaluationResult:
        """Verify a prediction against ground truth.

        AIME answers are integers from 0 to 999. Grading requires an
        explicitly marked answer (\\boxed{}, <answer> tags, or a final
        equation line); prose-only answers grade as PARSE_ERROR.

        Args:
            prediction: Model's full response text
            ground_truth: Correct answer

        Returns:
            EvaluationResult enum: CORRECT, INCORRECT, or PARSE_ERROR
        """
        grade = grade_aime_response(prediction, ground_truth)

        if grade.extracted_answer is None:
            return EvaluationResult.PARSE_ERROR
        return EvaluationResult.CORRECT if grade.is_correct else EvaluationResult.INCORRECT
