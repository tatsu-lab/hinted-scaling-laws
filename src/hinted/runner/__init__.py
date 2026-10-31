"""Shared machinery for experiment scripts: cells to tasks, task keys and resumable JSONL output.

`run_sweep` runs a grid of (model, benchmark, hint level) cells through the query engine, model
by model, appending each row as it completes; rerunning skips task keys already on disk.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hinted.benchmarks import (
    Benchmark,
    BenchmarkProblem,
    HintType,
    mask_word,
    truncate_word,
)
from hinted.models import ModelConfig
from hinted.query_engine import (
    InferenceConfig,
    InferenceTask,
    QueryResult,
    ResourcesConfig,
    run_inference,
)

# ---------------------------------------------------------------------------
# Task keys
# ---------------------------------------------------------------------------

_KEY_SEP = "|"


@dataclass(frozen=True)
class TaskKey:
    model_name: str
    problem_id: str
    hint_level: float
    sample_id: int

    def encode(self) -> str:
        return _KEY_SEP.join(
            [
                self.model_name,
                self.problem_id,
                f"h{self.hint_level:.2f}",
                f"s{self.sample_id}",
            ]
        )


def parse_task_key(key: str) -> TaskKey:
    model_name, problem_id, h_part, s_part = key.split(_KEY_SEP)
    return TaskKey(
        model_name=model_name,
        problem_id=problem_id,
        hint_level=float(h_part.removeprefix("h")),
        sample_id=int(s_part.removeprefix("s")),
    )


# ---------------------------------------------------------------------------
# Cell -> prompts
# ---------------------------------------------------------------------------


def get_cell_prompts(
    model_config: ModelConfig,
    benchmark: Benchmark,
    hint_level: float,
    hint_type: HintType = HintType.LOGIT_MIXING,
    *,
    samples_per_problem: int,
    hint_file: Optional[str] = None,
    problems: Optional[List[BenchmarkProblem]] = None,
    limit: Optional[int] = None,
) -> List[InferenceTask]:
    """Build every InferenceTask for one (model, benchmark, hint level) cell.

    Sample i uses hint rollout i (mod the number of loaded rollouts), so
    samples marginalize over hint variation deterministically. At
    hint_level == 0 tasks carry no teacher prompt and route to the plain
    vLLM worker.

    Hint types:
    - LOGIT_MIXING: the full hint goes to the teacher prompt and hint_level
      is the decode-time mixing weight; tasks carry a teacher_prompt and
      route to the joint-decode worker.
    - MASK: hint_level is the fraction of hint WORDS left visible, chosen
      by an RNG seeded by (problem, hint rollout, level), so every sample
      that uses the same hint rollout sees the same mask (on GSM8K, with one
      solution per problem, all samples at a level share it); the masked
      hint is placed inline in a single prompt (no teacher_prompt), so tasks
      route to the plain vLLM worker.
    - TRUNCATE: as MASK, but the visible fraction is the hint's PREFIX rather
      than a random subset, so nothing is replaced with [MASK] and no seed is
      involved.

    Pass `problems` (from benchmark.load_with_hint) to reuse one load across
    many cells; otherwise this loads from `hint_file` (required for
    hint_level > 0).
    """
    if hint_type not in (HintType.LOGIT_MIXING, HintType.MASK, HintType.TRUNCATE):
        raise NotImplementedError(f"hint_type {hint_type.value} not implemented")

    if problems is None:
        if hint_level > 0.0:
            if hint_file is None:
                raise ValueError("hint_file (or problems) required for hint_level > 0")
            problems = benchmark.load_with_hint(hint_file, limit=limit)
        else:
            problems = benchmark.load(limit=limit)

    tasks: List[InferenceTask] = []
    for problem in problems:
        for sample_id in range(samples_per_problem):
            key = TaskKey(
                model_name=model_config.name,
                problem_id=problem.problem_id,
                hint_level=hint_level,
                sample_id=sample_id,
            ).encode()
            if hint_level == 0.0:
                task = InferenceTask(task_key=key, prompt=problem.question_prompt())
            elif hint_type == HintType.LOGIT_MIXING:
                assert problem.hint is not None, f"{problem.problem_id} has no hints"
                task = InferenceTask(
                    task_key=key,
                    prompt=problem.question_prompt(),
                    teacher_prompt=problem.hinted_prompt(sample_id % len(problem.hint)),
                    hint_level=hint_level,
                )
            else:  # MASK / TRUNCATE: fractioned hint inline, single-model
                assert problem.hint is not None, f"{problem.problem_id} has no hints"
                rollout = sample_id % len(problem.hint)
                if hint_type == HintType.MASK:
                    text = mask_word(
                        problem.hint[rollout],
                        hint_level,
                        seed=f"{problem.problem_id}|{rollout}",
                        token=getattr(problem, "mask_token", "[MASK]"),
                    )
                else:  # HintType.TRUNCATE -- deterministic, so no seed
                    text = truncate_word(problem.hint[rollout], hint_level)
                task = InferenceTask(
                    task_key=key,
                    prompt=problem.hinted_prompt_at_level(text, hint_level),
                    hint_level=hint_level,
                )
            tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Result sink (incremental JSONL, resume by task key)
# ---------------------------------------------------------------------------


def load_completed_task_keys(path: Path) -> set[str]:
    """Task keys already recorded in a rollout file (for resume)."""
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                keys.add(json.loads(line)["task_key"])
    return keys


def result_row(
    result: QueryResult,
    task: InferenceTask,
    benchmark: Benchmark,
    answers: Dict[str, str],
    *,
    grade_inline: bool = True,
) -> dict:
    """One output row. grade_inline=False writes grade=None and never calls
    the benchmark's grader, for callers that grade in a separate pass."""
    key = parse_task_key(result.task_key)
    answer = answers[key.problem_id]
    grade = (
        None
        if not grade_inline or result.is_error or not benchmark.grades_inline
        else benchmark.verify(result.response_text, answer).value
    )
    return {
        "model": key.model_name,
        "problem_id": key.problem_id,
        "hint_level": key.hint_level,
        "sample_id": key.sample_id,
        "answer": answer,
        "grade": grade,
        # Raw prompts as built (pre chat template, which is applied
        # worker-side) — sanity-checkable and self-contained.
        "prompt": task.prompt,
        "teacher_prompt": task.teacher_prompt,
        **result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _has_weights(snapshot_dir: str) -> bool:
    """True if the snapshot dir holds a real weight file (not just metadata).

    The offline completeness check (snapshot_download local_files_only) is NOT
    sufficient on its own: a stale partial download (e.g. a tokenizer-only
    fetch) passes it, because offline HF only knows about files it previously
    fetched metadata for -- it has no record the weights should exist. So we
    also confirm an actual weight shard is present on disk, following symlinks
    to the blob (a dangling symlink or pointer stub fails the size floor).
    """
    from pathlib import Path

    p = Path(snapshot_dir)
    if not (p / "config.json").exists():
        return False
    weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    return any(w.exists() and w.stat().st_size > 1_000_000 for w in weights)


def _ensure_cached(hf_path: str, attempts: int = 5) -> None:
    """Download a model to the shared HF cache, retrying transient Hub errors.

    Raises if the cache can't be populated after `attempts` tries -- callers
    must not launch HF_HUB_OFFLINE=1 workers against a missing model, or the
    whole fleet crash-loops on the cache miss.
    """
    import time

    from huggingface_hub import snapshot_download
    from pathlib import Path

    # Only the files an inference engine actually loads. Two reasons:
    # gated repos (meta-llama/*) 403 on incidentals like .gitattributes even
    # when the weights are granted, and Llama repos ship a redundant
    # original/consolidated.*.pth that doubles the download. The offline
    # check below must use the SAME patterns, or a snapshot fetched this way
    # reads as incomplete and falls through to a re-download.
    patterns = ["*.safetensors", "*.safetensors.index.json", "*.bin",
                "*.bin.index.json", "config.json", "generation_config.json",
                "tokenizer*", "special_tokens_map.json", "*.model", "*.py"]

    # A local checkpoint directory (e.g. a converted DataDecide model) needs
    # no cache at all -- vLLM and the tokenizer load it by path.
    if Path(hf_path).is_dir():
        if not _has_weights(hf_path):
            raise RuntimeError(f"{hf_path} is a directory but holds no weights")
        print(f"[{hf_path}] local checkpoint directory", flush=True)
        return

    # Fast path: the snapshot resolves offline AND real weights are on disk.
    # Both checks are needed -- see _has_weights for why local_files_only
    # alone wrongly passes a stale metadata-only cache.
    try:
        path = snapshot_download(hf_path, local_files_only=True,
                                 allow_patterns=patterns)
        if _has_weights(path):
            print(f"[{hf_path}] already fully cached", flush=True)
            return
        print(f"[{hf_path}] cache present but weights missing; downloading", flush=True)
    except Exception:
        pass

    for attempt in range(1, attempts + 1):
        try:
            print(f"[{hf_path}] downloading weights to the local HF cache", flush=True)
            path = snapshot_download(hf_path, allow_patterns=patterns)
            if not _has_weights(path):
                raise RuntimeError("download completed but no weight shard on disk")
            return
        except Exception as e:
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(
                f"[{hf_path}] pre-download attempt {attempt}/{attempts} failed: {e}",
                flush=True,
            )
            if attempt == attempts:
                raise RuntimeError(
                    f"could not populate HF cache for {hf_path} after {attempts} "
                    "attempts; refusing to launch offline workers"
                ) from e
            time.sleep(wait)


def run_sweep(
    models: List[ModelConfig],
    benchmark: Benchmark,
    hint_levels: List[float],
    *,
    samples_per_problem: int,
    hint_file: Optional[str] = None,
    out_dir: str,
    resources: ResourcesConfig,
    hint_type: HintType = HintType.LOGIT_MIXING,
    make_inference_config: Optional[Callable[[ModelConfig], InferenceConfig]] = None,
    timeout_per_batch: Optional[float] = None,
    limit: Optional[int] = None,
) -> None:
    """Run the full (models x hint_levels x problems x samples) grid.

    Sequential over models (each model needs its own worker fleet). Per
    model, hint-free and hinted tasks run as separate batches (different
    worker kinds); every completed task is appended to
    out_dir/<model>.jsonl as it finishes, and rerunning skips completed
    task keys, so interrupted sweeps resume where they left off.

    hint_file may be omitted only for a hint-free sweep (all hint levels 0),
    e.g. a capability eval; the benchmark then loads without hints.
    """
    if hint_file is None and any(h != 0.0 for h in hint_levels):
        raise ValueError("hint_file is required when any hint_level > 0")
    if make_inference_config is None:
        make_inference_config = lambda model: InferenceConfig(model=model)  # noqa: E731

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sync_meta = Path(".sync_meta")
    if sync_meta.exists():
        shutil.copy(sync_meta, out_path / "sync_meta.txt")

    if hint_file is None:
        problems = benchmark.load(limit=limit)
    else:
        problems = benchmark.load_with_hint(hint_file, limit=limit)
    answers = {p.problem_id: p.answer for p in problems}

    for model in models:
        # Pre-download weights once from the driver: workers run with
        # HF_HUB_OFFLINE=1 (a 100+ worker fleet booting at once slams the
        # Hub API into 429 rate limits and death-spirals the fleet). This
        # MUST succeed before launching workers -- an offline worker on a
        # cache miss crash-loops the whole fleet. Retry transient Hub errors
        # (504s, Xet hiccups) and raise if the cache can't be populated.
        _ensure_cached(model.hf_path)
        # a joint-decode arm with a fixed external teacher needs side B's
        # weights cached too, for the same reason
        teacher = getattr(make_inference_config(model), "teacher_model", None)
        if teacher is not None and teacher.hf_path != model.hf_path:
            _ensure_cached(teacher.hf_path)

        model_out = out_path / f"{model.name}.jsonl"
        completed = load_completed_task_keys(model_out)

        tasks: List[InferenceTask] = []
        for hint_level in hint_levels:
            tasks.extend(
                get_cell_prompts(
                    model,
                    benchmark,
                    hint_level,
                    hint_type,
                    samples_per_problem=samples_per_problem,
                    problems=problems,
                )
            )
        todo = [t for t in tasks if t.task_key not in completed]
        print(
            f"[{model.name}] {len(tasks)} tasks total, "
            f"{len(completed)} already done, {len(todo)} to run",
            flush=True,
        )
        if not todo:
            continue

        vanilla = [t for t in todo if t.teacher_prompt is None]
        # Sort hinted tasks by (problem, sample) so tasks sharing prompt
        # prefixes (same problem; same teacher hint across hint levels) land
        # in the same worker batches and hit vLLM's prefix cache.
        joint = sorted(
            (t for t in todo if t.teacher_prompt is not None),
            key=lambda t: (
                parse_task_key(t.task_key).problem_id,
                parse_task_key(t.task_key).sample_id,
            ),
        )

        inference_config = make_inference_config(model)
        tasks_by_key = {t.task_key: t for t in todo}
        with model_out.open("a") as sink:

            def write_row(result: QueryResult) -> None:
                task = tasks_by_key[result.task_key]
                sink.write(
                    json.dumps(result_row(result, task, benchmark, answers)) + "\n"
                )
                sink.flush()

            for label, batch in (("vanilla", vanilla), ("joint", joint)):
                if not batch:
                    continue
                print(f"[{model.name}] running {label} batch: {len(batch)} tasks", flush=True)
                run_inference(
                    batch,
                    inference_config,
                    resources,
                    timeout=timeout_per_batch,
                    on_result=write_row,
                )
