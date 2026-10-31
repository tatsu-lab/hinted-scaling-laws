"""HLE (Humanity's Last Exam) benchmark loader.

The ``exactMatch`` open-answer subset (176 problems). Rollouts are graded
out-of-band by an LLM judge (``hinted.benchmarks.hle_grader``), not inline,
so this benchmark sets ``grades_inline = False``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hinted.benchmarks import Benchmark, BenchmarkProblem, EvaluationResult


@dataclass
class HLEProblem(BenchmarkProblem):
    """An HLE problem with HLE prompt templates.

    The prompt strings are the ones every HLE result in the paper used, kept
    byte-for-byte so that new rollouts stay directly comparable.
    ``Answer: {your answer}`` is a literal instruction in the prompt, not a
    format placeholder.
    """

    def question_prompt(self) -> str:
        return (
            f"Problem:\n{self.question.strip()}\n\n"
            "Give your final answer in the following format:\nAnswer: {your answer}"
        )

    def hinted_prompt_from_text(self, hint_text: str) -> str:
        return (
            "You will be given a problem and a hint to the problem. Use the hint if helpful.\n\n"
            f"Problem:\n{self.question.strip()}\n\n"
            f"Hint:\n{hint_text.strip()}\n\n"
            "Give your final answer in the following format:\nAnswer: {your answer}"
        )


class HLESubsetBenchmark(Benchmark):
    """
    A subset of the HLE benchmark that we filter down
    for having a correct rollout for and some other
    things like getting rid of multiple choice.
    """

    # HLE is judged by an LLM out-of-band; the sweep writes rollouts ungraded.
    grades_inline = False

    def __init__(self, data_path: str = "data/benchmarks/hle.jsonl") -> None:
        """Initialize HLE benchmark.

        Args:
            data_path: Path to the problems JSONL file (rows with
                problem_id, question, answer, source).
        """
        super().__init__(name="hle_subset")
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> List[BenchmarkProblem]:
        """Load HLE problems from disk.

        Args:
            limit: Optional limit on number of problems to load

        Returns:
            List of benchmark problems
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"HLE data not found at {self.data_path}. "
                "Format: JSONL rows with problem_id, question, answer, source."
            )

        with open(self.data_path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if limit is not None:
            rows = rows[:limit]

        return [
            HLEProblem(
                problem_id=row["problem_id"],
                question=row["question"],
                answer=str(row["answer"]),
                metadata={"source": row.get("source"), "problem_number": idx + 1},
            )
            for idx, row in enumerate(rows)
        ]

    def verify(self, prediction: str, ground_truth: str) -> EvaluationResult:
        """Not supported inline — HLE is graded out-of-band by an LLM judge.

        HLE correctness comes from an LLM judge (see ``hle_grader``) that needs
        the *question* as well as the response and gold answer, and makes a
        remote API call. The sweep's inline ``verify(prediction, ground_truth)``
        path has neither the question nor a place for per-rollout API calls, so
        HLE rollouts are written ungraded and scored later by a dedicated
        grading pass over the saved JSONL.
        """
        raise NotImplementedError(
            "HLE is graded out-of-band by the LLM-judge pass, not inline via "
            "verify(); see hinted.benchmarks.hle_grader.grade_hle_response."
        )
