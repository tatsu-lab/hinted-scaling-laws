from __future__ import annotations

from pathlib import Path

import pytest

from hinted.benchmarks import EvaluationResult
from hinted.benchmarks.gsm8k_delphi import GSM8KDelphiBenchmark, strict_extract

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("text, expected", [
    ("so 18 eggs.\n#### 18\n", "18"),
    ("#### $460\n", "460"),
    ("####   26", "26"),
    ("#### 1,430.", "1430"),
    ("#### 12\nQuestion: next one\n#### 99", "12"),   # the FIRST ####
    ("the answer is 18", None),                       # no #### at all
])
def test_strict_extract(text: str, expected: str | None) -> None:
    assert strict_extract(text) == expected


def test_verify_is_numeric_and_strict() -> None:
    bench = GSM8KDelphiBenchmark(data_path=str(REPO / "data/benchmarks/gsm8k_delphi.jsonl"))
    assert bench.verify("#### 12.00", "12") == EvaluationResult.CORRECT
    assert bench.verify("#### 2,125", "2125") == EvaluationResult.CORRECT
    assert bench.verify("#### 13", "12") == EvaluationResult.INCORRECT
    assert bench.verify("so it is 12", "12") == EvaluationResult.PARSE_ERROR


def test_teacher_prompt_holds_whole_solution_and_answer_line() -> None:
    bench = GSM8KDelphiBenchmark(data_path=str(REPO / "data/benchmarks/gsm8k_delphi.jsonl"),
                                 redact=False, teacher=True)
    problems = bench.load_with_hint(str(REPO / "data/benchmarks/gsm8k_solution_hints.jsonl"),
                                    limit=3)
    p = problems[0]
    teacher = p.hinted_prompt(0)
    assert teacher.startswith("<|start_header_id|>system<|end_header_id|>")
    assert p.question in teacher
    assert f"Answer: {p.hint[0]}\n#### {p.metadata['answer_line']}<|eot_id|>" in teacher
    assert teacher.endswith("<|start_header_id|>assistant<|end_header_id|>\n\nAnswer:")
    # the model itself reads the plain few-shot question
    assert p.question_prompt().endswith(f"Question: {p.question}\nAnswer:")
