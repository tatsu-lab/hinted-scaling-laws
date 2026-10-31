from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

import hinted.runner as runner
from hinted.benchmarks import HintType
from hinted.benchmarks.aime import AIMEBenchmark
from hinted.models import MODELS
from hinted.query_engine import InferenceTask, QueryResult, ResourcesConfig
from hinted.runner import (
    TaskKey,
    get_cell_prompts,
    load_completed_task_keys,
    parse_task_key,
    run_sweep,
)

MODEL = MODELS["Qwen3-0.6B"]


@pytest.fixture
def benchmark_files(tmp_path: Path) -> tuple[Path, Path]:
    problems = [
        {"problem_id": f"p{i}", "question": f"What is {i}+{i}?", "answer": 2 * i,
         "source": "MathArena/aime_2025"}
        for i in range(1, 4)
    ]
    hints = [
        {"problem_id": p["problem_id"], "rollout_id": r,
         "full_hint": f"Hint r{r} for {p['problem_id']}: double it."}
        for p in problems
        for r in range(2)  # fewer rollouts than samples, to exercise the mod
    ]
    problems_path = tmp_path / "problems.jsonl"
    hints_path = tmp_path / "hints.jsonl"
    problems_path.write_text("\n".join(json.dumps(p) for p in problems) + "\n")
    hints_path.write_text("\n".join(json.dumps(h) for h in hints) + "\n")
    return problems_path, hints_path


def test_task_key_round_trip() -> None:
    key = TaskKey(model_name="Qwen3-0.6B", problem_id="p1", hint_level=0.5, sample_id=3)
    encoded = key.encode()
    assert encoded == "Qwen3-0.6B|p1|h0.50|s3"
    assert parse_task_key(encoded) == key


def test_get_cell_prompts_hint_free(benchmark_files: tuple[Path, Path]) -> None:
    problems_path, _ = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))

    tasks = get_cell_prompts(MODEL, benchmark, 0.0, samples_per_problem=4)

    assert len(tasks) == 3 * 4
    assert len({t.task_key for t in tasks}) == len(tasks)
    assert all(t.teacher_prompt is None and t.hint_level == 0.0 for t in tasks)
    assert "What is 1+1?" in tasks[0].prompt


def test_get_cell_prompts_hinted_pairs_sample_to_rollout(
    benchmark_files: tuple[Path, Path],
) -> None:
    problems_path, hints_path = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))

    tasks = get_cell_prompts(
        MODEL, benchmark, 0.4, samples_per_problem=4, hint_file=str(hints_path)
    )

    assert len(tasks) == 3 * 4
    assert all(t.hint_level == 0.4 and t.teacher_prompt is not None for t in tasks)
    p1 = {parse_task_key(t.task_key).sample_id: t for t in tasks
          if parse_task_key(t.task_key).problem_id == "p1"}
    assert "Hint r0" in p1[0].teacher_prompt
    assert "Hint r1" in p1[1].teacher_prompt
    assert "Hint r0" in p1[2].teacher_prompt  # sample 2 wraps to rollout 0


def test_get_cell_prompts_mask_is_inline_and_vanilla(
    benchmark_files: tuple[Path, Path],
) -> None:
    problems_path, hints_path = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))

    tasks = get_cell_prompts(
        MODEL, benchmark, 0.5, HintType.MASK,
        samples_per_problem=4, hint_file=str(hints_path),
    )

    assert len(tasks) == 3 * 4
    # Mask hinting is single-model: no teacher prompt (routes to vanilla),
    # and the (partially masked) hint is inline in the student prompt.
    assert all(t.teacher_prompt is None and t.hint_level == 0.5 for t in tasks)
    assert all("Hint:" in t.prompt and "[MASK]" in t.prompt for t in tasks)
    p1 = {parse_task_key(t.task_key).sample_id: t for t in tasks
          if parse_task_key(t.task_key).problem_id == "p1"}
    # sample i uses hint rollout i mod n_rollouts; masking is deterministic,
    # so sample 2 (wraps to rollout 0) reproduces sample 0's prompt exactly,
    # while rollout 1 (sample 1) differs.
    assert p1[2].prompt == p1[0].prompt
    assert p1[1].prompt != p1[0].prompt


def test_get_cell_prompts_requires_hint_source(benchmark_files: tuple[Path, Path]) -> None:
    problems_path, _ = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))
    with pytest.raises(ValueError, match="hint_file"):
        get_cell_prompts(MODEL, benchmark, 0.5, samples_per_problem=1)


def test_result_row_records_error_and_skips_grading(
    benchmark_files: tuple[Path, Path],
) -> None:
    problems_path, _ = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))
    failed = QueryResult(
        task_key="Qwen3-0.6B|p1|h0.50|s0",
        response_text="",
        finish_reason="missing",
        is_error=True,
        error="no result returned (timeout or worker failure)",
    )
    task = InferenceTask(
        task_key="Qwen3-0.6B|p1|h0.50|s0",
        prompt="student prompt",
        teacher_prompt="teacher prompt",
        hint_level=0.5,
    )

    row = runner.result_row(failed, task, benchmark, {"p1": "2"})

    assert row["grade"] is None
    assert row["is_error"] is True
    assert row["error"] == "no result returned (timeout or worker failure)"
    assert row["answer"] == "2"
    assert row["prompt"] == "student prompt"
    assert row["teacher_prompt"] == "teacher prompt"


def _fake_run_inference(record: List[List[InferenceTask]]):
    """Stub returning '\\boxed{<n+n>}' for pN prompts, wired through on_result."""

    def fake(tasks, inference_config, resources, timeout=None, on_result=None):
        record.append(list(tasks))
        results = []
        for t in tasks:
            n = int(parse_task_key(t.task_key).problem_id.removeprefix("p"))
            result = QueryResult(
                task_key=t.task_key,
                response_text=f"Reasoning... \\boxed{{{n + n}}}",
                finish_reason="stop",
                is_error=False,
            )
            if on_result is not None:
                on_result(result)
            results.append(result)
        return results

    return fake


def test_run_sweep_writes_grades_and_resumes(
    benchmark_files: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems_path, hints_path = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))
    out_dir = tmp_path / "out"
    resources = ResourcesConfig(log_dir=str(tmp_path / "slurm"))

    calls: List[List[InferenceTask]] = []
    monkeypatch.setattr(runner, "run_inference", _fake_run_inference(calls))
    monkeypatch.setattr(runner, "_ensure_cached", lambda *a, **k: None)

    kwargs = dict(
        models=[MODEL],
        benchmark=benchmark,
        hint_levels=[0.0, 0.5],
        samples_per_problem=2,
        hint_file=str(hints_path),
        out_dir=str(out_dir),
        resources=resources,
    )
    run_sweep(**kwargs)

    # Two batches: vanilla (h=0) then joint (h=0.5), 6 tasks each.
    assert [len(c) for c in calls] == [6, 6]
    assert all(t.teacher_prompt is None for t in calls[0])
    assert all(t.teacher_prompt is not None for t in calls[1])

    rows = [json.loads(line) for line in (out_dir / f"{MODEL.name}.jsonl").open()]
    assert len(rows) == 12
    assert all(row["grade"] == "correct" for row in rows)  # boxed answer matches
    assert {row["hint_level"] for row in rows} == {0.0, 0.5}
    assert rows[0]["model"] == MODEL.name
    # Rows are self-contained: prompts recorded (teacher only for hinted rows).
    assert all("Problem:" in row["prompt"] for row in rows)
    for row in rows:
        if row["hint_level"] == 0.0:
            assert row["teacher_prompt"] is None
        else:
            assert "Hint" in row["teacher_prompt"]

    completed = load_completed_task_keys(out_dir / f"{MODEL.name}.jsonl")
    assert len(completed) == 12

    # Rerunning is a no-op: everything is already recorded.
    run_sweep(**kwargs)
    assert len(calls) == 2  # no new run_inference calls
    rows_after = [json.loads(line) for line in (out_dir / f"{MODEL.name}.jsonl").open()]
    assert len(rows_after) == 12


def test_run_sweep_hint_free_without_hint_file(
    benchmark_files: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems_path, _ = benchmark_files
    benchmark = AIMEBenchmark(data_path=str(problems_path))
    out_dir = tmp_path / "out"

    calls: List[List[InferenceTask]] = []
    monkeypatch.setattr(runner, "run_inference", _fake_run_inference(calls))
    monkeypatch.setattr(runner, "_ensure_cached", lambda *a, **k: None)

    run_sweep(
        models=[MODEL],
        benchmark=benchmark,
        hint_levels=[0.0],
        samples_per_problem=2,
        out_dir=str(out_dir),
        resources=ResourcesConfig(log_dir=str(tmp_path / "slurm")),
    )

    assert [len(c) for c in calls] == [6]  # one vanilla batch, no joint batch
    assert all(t.teacher_prompt is None for t in calls[0])
    rows = [json.loads(line) for line in (out_dir / f"{MODEL.name}.jsonl").open()]
    assert len(rows) == 6


def test_has_weights(tmp_path: Path) -> None:
    from hinted.runner import _has_weights

    # Metadata-only cache (the stale-tokenizer failure mode): must be False.
    meta_only = tmp_path / "meta"
    meta_only.mkdir()
    (meta_only / "config.json").write_text("{}")
    (meta_only / "tokenizer.json").write_text("{}")
    assert _has_weights(str(meta_only)) is False

    # Real weight shard present: True.
    full = tmp_path / "full"
    full.mkdir()
    (full / "config.json").write_text("{}")
    (full / "model.safetensors").write_bytes(b"\0" * 2_000_000)
    assert _has_weights(str(full)) is True

    # A tiny pointer/stub weight file fails the size floor.
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "config.json").write_text("{}")
    (stub / "model.safetensors").write_bytes(b"\0" * 100)
    assert _has_weights(str(stub)) is False


def test_run_sweep_requires_hint_file_for_hinted_levels(
    benchmark_files: tuple[Path, Path], tmp_path: Path
) -> None:
    problems_path, _ = benchmark_files
    with pytest.raises(ValueError, match="hint_file is required"):
        run_sweep(
            models=[MODEL],
            benchmark=AIMEBenchmark(data_path=str(problems_path)),
            hint_levels=[0.0, 0.5],
            samples_per_problem=1,
            out_dir=str(tmp_path / "out"),
            resources=ResourcesConfig(log_dir=str(tmp_path / "slurm")),
        )
