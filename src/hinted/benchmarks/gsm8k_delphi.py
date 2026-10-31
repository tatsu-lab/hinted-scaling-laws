"""GSM8K for the Delphi ladder: five-shot raw completion, hinted by the problem's own solution.

H = 0 is the plain few-shot prompt, with the exemplars stored per problem in the data file. The
runner uses the hint-block format (``hint_block=True``): the degraded solution goes on a
``Hint:`` line, the exemplars show hints degraded the same way, and the model writes a full
solution. ``degradation`` picks truncation or masking for the exemplar hints, and ``redact``
masks the gold answer in the visible hint text (on for truncation, off for masking).
``teacher=True`` builds the logit-mixing teacher's Llama-3.1 chat prompt instead.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hinted.benchmarks import (
    Benchmark, BenchmarkProblem, EvaluationResult, mask_word, truncate_word,
)

REDACTION = "<mask>"
# like lm-eval-harness gsm8k "strict-match" (the number after "#### "), but allowing any
# whitespace after "####" and a "$" in the number.
_STRICT = re.compile(r"####\s*(-?[$0-9.,]+)")
# The logit-mixing teacher's prompt (Llama-3.1-8B-Instruct's chat format, with
# the template's default system header): the teacher has answered the question
# with the reference solution, and is asked for another solution.
TEACHER_PROMPT = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{question}\n\nEnd your answer with: #### <number><|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Answer: {solution}\n#### {answer_line}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Great. Can you solve it another way?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\nAnswer:"
)


def redact_answer(text: str, answer: str) -> str:
    """Replace standalone occurrences of `answer` in `text` with <mask>.

    Standalone means not embedded in a longer number: 18 must not match
    inside 180 or 1.18, hence the (?<![\\d.]) / (?![\\d.]) guards. Both the
    plain and the comma-grouped spelling of the answer are covered (1000 and
    1,000), since GSM8K solutions write large numbers both ways. Word forms
    ("eighteen") are deliberately not covered — see the design doc.
    """
    plain = answer.strip().replace(",", "").replace("$", "")
    variants = {plain}
    sign, digits = (plain[0], plain[1:]) if plain.startswith("-") else ("", plain)
    if digits.isdigit() and len(digits) > 3:
        grouped = ""
        for i, ch in enumerate(reversed(digits)):
            if i and i % 3 == 0:
                grouped = "," + grouped
            grouped = ch + grouped
        variants.add(sign + grouped)
    pattern = "|".join(re.escape(v) for v in sorted(variants, key=len,
                                                    reverse=True))
    return re.sub(rf"(?<![\d.])(?:{pattern})(?![\d.])", REDACTION, text)


def strict_extract(completion: str) -> Optional[str]:
    """The number after the first ``####``, normalised; None without one."""
    m = _STRICT.search(completion)
    if m is None:
        return None
    return m.group(1).replace("$", "").replace(",", "").strip().rstrip(".")


def parse_exemplars(prefix: str) -> list[tuple[str, str, str]]:
    """The exemplar prefix as (question, working, answer) triples.

    Splitting is validated by reassembly: the triples rebuilt into blocks
    must reproduce the stored prefix byte-for-byte, so any format surprise
    fails at load rather than silently building wrong prompts.
    """
    # split at block boundaries, not on every blank line: some workings
    # contain blank lines internally
    blocks = re.split(r"(?<=\n\n)(?=Question: )", prefix)
    triples = []
    for b in blocks:
        q, _, rest = b.partition("\nAnswer: ")
        assert q.startswith("Question: "), f"bad block start: {b[:40]!r}"
        working, _, tail = rest.rpartition("\n#### ")
        ans = tail.rstrip("\n")
        rebuilt = f"{q}\nAnswer: {working}\n#### {ans}\n\n"
        assert rebuilt == b, f"block reassembly mismatch: {b[:60]!r}"
        triples.append((q[len("Question: "):], working, ans))
    assert len(triples) == 5, f"expected 5 exemplars, got {len(triples)}"
    return triples


@dataclass
class GSM8KDelphiProblem(BenchmarkProblem):
    """One GSM8K problem carrying its snapshotted 5-shot exemplar prefix."""

    redact: bool = True
    hint_block: bool = False
    degradation: str = "truncate"
    teacher: bool = False
    mask_token = "<mask>"          # picked up by the runner's MASK path

    def question_prompt(self) -> str:
        # the ladder's standard prompt: ends "Answer:" with no trailing
        # space or newline
        return (f"{self.metadata['exemplar_prefix']}"
                f"Question: {self.question}\nAnswer:")

    def hinted_prompt_from_text(self, hint_text: str) -> str:
        if self.teacher:
            return TEACHER_PROMPT.format(question=self.question, solution=hint_text,
                                         answer_line=self.metadata["answer_line"])
        # the hinted target block: "Answer: " then the (truncated) solution,
        # then a trailing newline. Redaction comes after truncation, so the
        # word count the runner's truncate_word sees is the un-redacted
        # solution's and the cut never shifts.
        shown = (redact_answer(hint_text, self.answer) if self.redact
                 else hint_text)
        return (f"{self.metadata['exemplar_prefix']}"
                f"Question: {self.question}\nAnswer: {shown}\n")

    def hinted_prompt_at_level(self, hint_text: str, level: float) -> str:
        """Continuation format: level-blind. Hint-block format: exemplars
        are dressed with hints truncated at the target's own level."""
        if not self.hint_block:
            return self.hinted_prompt_from_text(hint_text)
        assert level > 0.0, "h = 0 goes through question_prompt"
        parts = []
        for ei, (eq, ew, ea) in enumerate(
                parse_exemplars(self.metadata["exemplar_prefix"])):
            # same operator and order as the target: degrade the raw
            # working, then (if enabled) redact the fragment
            if self.degradation == "mask":
                frag = mask_word(ew, level,
                                 seed=f"{self.problem_id}|exemplar|{ei}",
                                 token=self.mask_token)
            else:
                frag = truncate_word(ew, level)
            if self.redact:
                frag = redact_answer(frag, ea)
            parts.append(f"Question: {eq}\nHint: {frag}\n"
                         f"Answer: {ew}\n#### {ea}\n\n")
        shown = (redact_answer(hint_text, self.answer) if self.redact
                 else hint_text)
        parts.append(f"Question: {self.question}\nHint: {shown}\nAnswer:")
        return "".join(parts)


class GSM8KDelphiBenchmark(Benchmark):
    """The 256-problem GSM8K subset in the Delphi few-shot completion format."""

    grades_inline = True

    def __init__(self, data_path: str = "data/benchmarks/gsm8k_delphi.jsonl",
                 redact: bool = True, hint_block: bool = False,
                 degradation: str = "truncate", teacher: bool = False) -> None:
        super().__init__(name="gsm8k_delphi")
        assert degradation in ("truncate", "mask")
        self.data_path = Path(data_path)
        self.redact = redact
        self.hint_block = hint_block
        self.degradation = degradation
        self.teacher = teacher

    def load(self, limit: Optional[int] = None) -> List[BenchmarkProblem]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"GSM8K-Delphi data not found at {self.data_path}. It ships "
                "as data/benchmarks/gsm8k_delphi.jsonl; run from the repository root."
            )
        with open(self.data_path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            rows = rows[:limit]
        return [
            GSM8KDelphiProblem(
                problem_id=row["problem_id"],
                question=row["question"],
                answer=str(row["answer"]),
                metadata=row["metadata"],
                redact=self.redact,
                hint_block=self.hint_block,
                degradation=self.degradation,
                teacher=self.teacher,
            )
            for row in rows
        ]

    def verify(self, prediction: str, ground_truth: str) -> EvaluationResult:
        extracted = strict_extract(prediction)
        if extracted is None:
            return EvaluationResult.PARSE_ERROR
        gold = ground_truth.replace(",", "").replace("$", "").strip()
        try:
            correct = float(extracted) == float(gold)
        except ValueError:
            correct = extracted == gold
        return EvaluationResult.CORRECT if correct else EvaluationResult.INCORRECT
