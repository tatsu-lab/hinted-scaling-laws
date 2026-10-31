"""Base classes for benchmarks."""

import hashlib
import json
import math
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationResult(Enum):
    """Result of evaluating a model prediction."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARSE_ERROR = "parse_error"  # Unable to parse model output

class HintType(Enum):

    TRUNCATE = "truncate"
    MASK = "mask"
    LOGIT_MIXING = "logit_mixing"


def _stable_int(*parts: object) -> int:
    """Process-stable hash (unlike hash()), for reproducible masking."""
    text = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def mask_word(full_hint: str, fraction: float, *, seed: str,
              token: str = "[MASK]") -> str:
    """Mask hint primitive.

    Keep ceil(len * fraction) whitespace-delimited words visible; replace the
    rest with the literal `token` (default ``[MASK]``, the HLE/AIME sweeps'
    convention; GSM8K-Delphi passes ``<mask>``), preserving whitespace.
    The masked subset is chosen by a hash-seeded RNG over (seed, fraction),
    so a given (seed, fraction) always yields the same mask and larger
    fractions are NOT nested subsets of smaller ones.
    fraction >= 1 returns the text unchanged; fraction <= 0 masks every word.
    """
    words = list(re.finditer(r"\S+", full_hint))
    keep = max(0, min(len(words), math.ceil(len(words) * fraction)))
    visible = set(range(len(words)))
    if len(words) > keep:
        rng = random.Random(_stable_int(seed, "mask_word", fraction))
        visible -= set(rng.sample(range(len(words)), len(words) - keep))

    parts: list[str] = []
    cursor = 0
    for idx, word in enumerate(words):
        parts.append(full_hint[cursor : word.start()])
        parts.append(word.group(0) if idx in visible else token)
        cursor = word.end()
    parts.append(full_hint[cursor:])
    return "".join(parts)


def truncate_word(full_hint: str, fraction: float) -> str:
    """Truncate hint primitive.

    Keep the FIRST ceil(len * fraction) whitespace-delimited words and drop the
    rest. The kept span is sliced out of the ORIGINAL string (up to the end of
    the last kept word), so interior whitespace, newlines and markdown survive
    untouched. fraction >= 1 returns the text unchanged; fraction <= 0 returns
    "". Unlike mask_word there is no RNG and so no seed: which words survive is
    fully determined by the fraction, and smaller fractions ARE prefixes of
    larger ones.
    """
    words = list(re.finditer(r"\S+", full_hint))
    keep = max(0, min(len(words), math.ceil(len(words) * fraction)))
    if keep == 0:
        return ""
    if keep >= len(words):
        return full_hint
    return full_hint[: words[keep - 1].end()]


@dataclass
class BenchmarkProblem(ABC):
    """A single problem from a benchmark.

    Prompt construction is benchmark-specific (answer-format instructions,
    hint framing), so subclasses implement the two prompt builders. We only
    support logit-mixing hinting for now: the hint level h is applied at
    decode time as the teacher's mixing weight, never by diluting the hint
    text — so the teacher prompt always contains the full hint.

    Attributes:
        problem_id: Unique identifier for the problem
        question: The problem statement/question
        answer: The correct answer
        metadata: Additional problem-specific metadata
        hint: Candidate hint texts for this problem, one per rollout
            (possibly a singleton), if loaded
        hint_result: Evaluation result of the hint (if verified)
    """

    problem_id: str
    question: str
    answer: str
    metadata: Dict[str, Any]
    hint: Optional[list[str]] = None
    hint_result: Optional["EvaluationResult"] = None

    @abstractmethod
    def question_prompt(self) -> str:
        """Build the hint-free prompt (what the base model sees)."""

    @abstractmethod
    def hinted_prompt_from_text(self, hint_text: str) -> str:
        """Wrap the given hint text in this benchmark's hinted-prompt framing.

        Used for both full-hint (logit-mixing teacher) and fractionally
        masked/truncated hints — the caller supplies whatever hint text
        should appear in the prompt.
        """

    def hinted_prompt_at_level(self, hint_text: str, level: float) -> str:
        """Level-aware hinted prompt; default framing ignores the level.

        The runner's MASK/TRUNCATE path calls this with the already
        fractioned hint text and the hint level. Most benchmarks frame a
        hint the same way at every level, so the default delegates to
        hinted_prompt_from_text; a benchmark whose PROMPT STRUCTURE depends
        on the level (e.g. GSM8K-Delphi's hint-block format, which dresses
        its few-shot exemplars to match the target's level) overrides this.
        """
        return self.hinted_prompt_from_text(hint_text)

    def hinted_prompt(self, hint_index: int = 0) -> str:
        """Prompt containing the full hint rollout `hint_index`.

        Raises:
            ValueError: If no hints are loaded on this problem
        """
        if self.hint is None:
            raise ValueError(f"Problem {self.problem_id} has no hint loaded")
        return self.hinted_prompt_from_text(self.hint[hint_index])


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    # Whether the sweep can grade each rollout inline via `verify()` as it is
    # written. True for deterministic graders (AIME integer, GSM8K numeric).
    # Benchmarks graded out-of-band (e.g. HLE's LLM judge, which needs the
    # question and makes a remote call) set this False; the sweep then writes
    # rollouts with grade=None for a later dedicated grading pass.
    grades_inline: bool = True

    def __init__(self, name: str) -> None:
        """Initialize benchmark.

        Args:
            name: Name of the benchmark
        """
        self.name = name

    @abstractmethod
    def load(
        self, 
        limit: Optional[int] = None
    ) -> List[BenchmarkProblem]:
        """Load benchmark problems.

        Args:
            limit: Optional limit on number of problems to load

        Returns:
            List of benchmark problems
        """
        pass

    @abstractmethod
    def verify(self, prediction: str, ground_truth: str) -> EvaluationResult:
        """Verify a prediction against ground truth.

        Args:
            prediction: Model's predicted answer
            ground_truth: Correct answer

        Returns:
            EvaluationResult enum indicating if correct, incorrect, or parse error
        """
        pass

    def load_with_hint(
        self, 
        hint_file: str,
        limit: Optional[int] = None, 
    ) -> List[BenchmarkProblem]:
        """Load benchmark problems with hints.

        Args:
            hint_file: Path to JSONL file of hint rows (e.g. data/benchmarks/aime_solution_hints.jsonl).
                Each row must have problem_id, rollout_id, and full_hint.
            limit: Optional limit on number of problems

        Returns:
            List of benchmark problems, each with all its hint rollouts loaded
        """

        assert hint_file is not None

        problems = self.load(limit=limit)

        with open(hint_file, "r") as f:
            hint_data = [json.loads(line) for line in f if line.strip()]

        # Collect every rollout for each problem, ordered by rollout_id
        hint_map: Dict[str, List[Any]] = defaultdict(list)
        for item in hint_data:
            hint_map[item["problem_id"]].append((item["rollout_id"], item["full_hint"]))

        for problem in problems:
            if problem.problem_id in hint_map:
                problem.hint = [h for _, h in sorted(hint_map[problem.problem_id])]
            else:
                raise ValueError("There is a problem with no hint")

        # Do some other checks here

        return problems


