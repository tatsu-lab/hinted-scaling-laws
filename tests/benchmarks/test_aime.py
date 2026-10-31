from __future__ import annotations

import json
from pathlib import Path

import pytest

from hinted.benchmarks.aime import AIMEBenchmark, AIMEProblem

REAL_DATA_PATH = Path("data/benchmarks/aime2025_2026.jsonl")

FAKE_ROWS = [
    {
        "problem_id": "aime2025_2026_0001",
        "question": "What is $1+1$?",
        "answer": 2,
        "source": "MathArena/aime_2025",
    },
    {
        "problem_id": "aime2025_2026_0002",
        "question": "What is $2+2$?",
        "answer": "4",
        "source": "MathArena/aime_2025",
    },
    {
        "problem_id": "aime2025_2026_0003",
        "question": "What is $3+3$?",
        "answer": 6,
        "source": "MathArena/aime_2026",
    },
]


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "aime_test.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in FAKE_ROWS) + "\n")
    return path


def test_load_maps_fields(data_file: Path) -> None:
    problems = AIMEBenchmark(data_path=str(data_file)).load()

    assert len(problems) == len(FAKE_ROWS)
    first = problems[0]
    assert first.problem_id == "aime2025_2026_0001"
    assert first.question == "What is $1+1$?"
    assert first.metadata == {"source": "MathArena/aime_2025", "problem_number": 1}
    assert first.hint is None
    assert first.hint_result is None


def test_load_coerces_answer_to_str(data_file: Path) -> None:
    problems = AIMEBenchmark(data_path=str(data_file)).load()

    # Answers appear as both ints and strings in JSON; both come out as str.
    assert [p.answer for p in problems] == ["2", "4", "6"]
    assert all(isinstance(p.answer, str) for p in problems)


def test_load_limit(data_file: Path) -> None:
    benchmark = AIMEBenchmark(data_path=str(data_file))

    assert len(benchmark.load(limit=2)) == 2
    assert len(benchmark.load(limit=len(FAKE_ROWS) + 100)) == len(FAKE_ROWS)
    assert len(benchmark.load(limit=None)) == len(FAKE_ROWS)


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blanks.jsonl"
    rows = "\n\n".join(json.dumps(row) for row in FAKE_ROWS)
    path.write_text(f"\n{rows}\n\n")

    problems = AIMEBenchmark(data_path=str(path)).load()

    assert len(problems) == len(FAKE_ROWS)


def test_load_problem_numbers_are_sequential(data_file: Path) -> None:
    problems = AIMEBenchmark(data_path=str(data_file)).load()

    assert [p.metadata["problem_number"] for p in problems] == [1, 2, 3]


def test_missing_file_raises() -> None:
    benchmark = AIMEBenchmark(data_path="does/not/exist.jsonl")

    with pytest.raises(FileNotFoundError, match="AIME data not found"):
        benchmark.load()


def test_benchmark_name() -> None:
    assert AIMEBenchmark(data_path="unused.jsonl").name == "aime2025_2026"


def _problem(hint: list[str] | None = None) -> AIMEProblem:
    return AIMEProblem(
        problem_id="p1",
        question="What is $1+1$?  ",
        answer="2",
        metadata={},
        hint=hint,
    )


def test_question_prompt() -> None:
    prompt = _problem().question_prompt()

    assert prompt == (
        "Put your final answer within \\boxed{}.\n"
        "The answer is an integer between 0 and 999 inclusive.\n\n"
        "Problem:\nWhat is $1+1$?"
    )


def test_hinted_prompt_uses_full_hint() -> None:
    prompt = _problem(hint=["Add the numbers.\n", "Count up."]).hinted_prompt()

    assert prompt == (
        "You will be given a problem and a hint to the problem.\n"
        "Put your final answer within \\boxed{}.\n"
        "The answer is an integer between 0 and 999 inclusive.\n\n"
        "Problem:\nWhat is $1+1$?\n\n"
        "Hint:\nAdd the numbers."
    )


def test_hinted_prompt_selects_rollout() -> None:
    prompt = _problem(hint=["Add the numbers.", "Count up."]).hinted_prompt(hint_index=1)

    assert prompt.endswith("Hint:\nCount up.")


def test_hinted_prompt_without_hint_raises() -> None:
    with pytest.raises(ValueError, match="no hint loaded"):
        _problem().hinted_prompt()


@pytest.mark.skipif(not REAL_DATA_PATH.exists(), reason="real AIME data only on cluster")
class TestRealData:
    def test_load_full_dataset(self) -> None:
        problems = AIMEBenchmark().load()

        assert len(problems) == 60
        assert len({p.problem_id for p in problems}) == 60
        assert all(p.answer.isdigit() for p in problems)
        sources = {p.metadata["source"] for p in problems}
        assert sources == {"MathArena/aime_2025", "MathArena/aime_2026"}

    def test_default_data_path(self) -> None:
        assert AIMEBenchmark().data_path == REAL_DATA_PATH
