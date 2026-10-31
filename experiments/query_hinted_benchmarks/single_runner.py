"""Run one hinted-benchmark cell: one model, benchmark and method over a set of hint levels.

Writes one ungraded JSONL row per rollout to --out and skips rollouts already there. Which
cells to run is `launch_all.py`'s job; see this folder's README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from hinted.benchmarks import Benchmark, HintType
from hinted.benchmarks.aime import AIMEBenchmark
from hinted.benchmarks.gsm8k_delphi import GSM8KDelphiBenchmark
from hinted.benchmarks.hle import HLESubsetBenchmark
from hinted.models import MODELS, ModelConfig
from hinted.query_engine.types import InferenceConfig
from hinted.query_engine.workers import (
    JointDecodeInferenceWorker,
    VLLMInferenceWorker,
)
from hinted.runner import (
    get_cell_prompts,
    load_completed_task_keys,
    result_row,
)

METHODS = ["no_hint", "truncate", "mask", "logit_mix", "logit_mix_std"]
MIXING = ("logit_mix", "logit_mix_std")

HINT_TYPE = {
    "truncate": HintType.TRUNCATE,
    "mask": HintType.MASK,
    "logit_mix": HintType.LOGIT_MIXING,
    "logit_mix_std": HintType.LOGIT_MIXING,
    "no_hint": HintType.TRUNCATE,  # unused: H = 0 takes the hint-free branch
}
# Logit mixing samples from the union of each side's top MIX_TOP_K tokens at
# the model's temperature, with no top-p, model top-k or repetition penalty.
MIX_TOP_K = 16


def make_benchmark(benchmark: str, method: str) -> Benchmark:
    """The benchmark object for this cell.

    Only GSM8K-Delphi varies by method, because its arms frame the hint
    differently:

    - truncate: the redacted reference solution in a labelled hint block,
      with a fresh answer written below it. The unhinted anchor is this
      arm's H = 0, so no_hint uses the same framing.
    - mask: the same hint block, word-masked and UN-redacted — answer
      visibility then scales smoothly with H instead of hitting
      truncation's 0.9 -> 1.0 leak cliff.
    - logit_mix: the model reads the plain question; the teacher reads the
      un-redacted solution in its own chat format.

    On AIME and HLE the logit-mixing teacher reads the benchmark's usual
    hinted prompt with the whole solution.
    """
    if benchmark == "aime":
        return AIMEBenchmark()
    if benchmark == "hle":
        return HLESubsetBenchmark()
    if method == "mask":
        return GSM8KDelphiBenchmark(redact=False, hint_block=True,
                                    degradation="mask")
    if method in MIXING:
        return GSM8KDelphiBenchmark(redact=False, teacher=True)
    return GSM8KDelphiBenchmark(redact=True, hint_block=True,
                                degradation="truncate")


def build_config(benchmark: str, model_name: str, method: str, *,
                 teacher: str | None, tensor_parallel: int,
                 joint_gpus: int) -> InferenceConfig:
    """Decoding settings, matching the runs that produced the paper."""
    model = MODELS[model_name]
    common: dict = {}
    if benchmark == "gsm8k":
        # The Delphi ladder is prompted as raw few-shot completion, so no
        # chat template (the teacher's prompt carries its own chat format).
        common["apply_chat_template"] = False
        common["stop"] = ("Question:", "</s>", "<|im_end|>")
        if method == "no_hint":
            model = replace(model, top_k=0)          # 0 turns top-k off in vLLM

    if method in MIXING:
        def mixing(m: ModelConfig) -> ModelConfig:
            return replace(m, top_k=MIX_TOP_K, top_p=1.0, repetition_penalty=1.0)
        return InferenceConfig(
            model=mixing(model),
            **common,
            teacher_model=mixing(MODELS[teacher]),
            joint_teacher_scale="std" if method == "logit_mix_std" else "none",
            joint_top_k_payload=MIX_TOP_K,
            joint_gpus_per_worker=joint_gpus,
            # Qwen3.5's joint engines hit a vLLM prefix-caching bug.
            joint_enable_prefix_caching=not model_name.startswith("Qwen3.5"),
        )
    return InferenceConfig(model=model, **common, tensor_parallel_size=tensor_parallel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="a model registry name")
    ap.add_argument("--benchmark", required=True,
                    choices=["gsm8k", "aime", "hle"])
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--levels", type=float, nargs="+", required=True,
                    help="hint levels for this cell; 0.0 for no_hint; the "
                         "teacher weights for logit mixing")
    ap.add_argument("--teacher", default=None,
                    help="logit mixing only: the teacher's registry name")
    ap.add_argument("--samples", type=int, required=True,
                    help="rollouts per problem")
    ap.add_argument("--hint-file", default=None,
                    help="reference solutions; required unless every level "
                         "is 0")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N problems only, for smoke tests")
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="vLLM tensor parallel size; single-engine methods only")
    ap.add_argument("--joint-gpus", type=int, default=2, choices=[1, 2],
                    help="GPUs for logit mixing: 2 = one engine each, "
                         "1 = share a card at 0.45 memory fraction each")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="tasks per engine call; bounds what an interruption "
                         "loses")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the work and exit without loading a model")
    args = ap.parse_args()

    load_dotenv()  # HF_TOKEN for gated weights

    if args.model not in MODELS:
        raise SystemExit(f"{args.model!r} is not in the model registry")
    if args.method in MIXING:
        if args.tensor_parallel != 1:
            raise SystemExit(
                "--tensor-parallel is not available for logit mixing: its two "
                "engines are each pinned to one device by the vendored joint "
                "decoder, so each engine's model must fit on a single GPU. Use "
                "--joint-gpus 2 and a larger card."
            )
        if args.teacher not in MODELS:
            raise SystemExit(f"logit mixing needs --teacher, a model registry "
                             f"name; got {args.teacher!r}")
    elif args.teacher:
        raise SystemExit("--teacher is for logit mixing only")
    hinted = [h for h in args.levels if h > 0]
    if hinted and not args.hint_file:
        raise SystemExit("--hint-file is required for hint levels above 0")
    if args.method != "no_hint" and len(hinted) != len(args.levels):
        raise SystemExit(
            f"{args.method} takes hint levels above 0 only, got {args.levels}; "
            "the H = 0 measurement is its own method, no_hint")
    if args.method == "no_hint" and hinted:
        raise SystemExit(
            f"no_hint takes level 0.0 only, got {args.levels}; use a hinting "
            "method for H > 0")

    benchmark = make_benchmark(args.benchmark, args.method)
    problems = (benchmark.load_with_hint(args.hint_file, limit=args.limit)
                if hinted else benchmark.load(limit=args.limit))
    answers = {p.problem_id: p.answer for p in problems}

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed_task_keys(out_file)

    tasks = []
    for level in args.levels:
        tasks += get_cell_prompts(
            MODELS[args.model], benchmark, level, HINT_TYPE[args.method],
            samples_per_problem=args.samples, problems=problems,
        )
    todo = [t for t in tasks if t.task_key not in done]

    print(f"{args.model} | {args.benchmark} | {args.method} | "
          f"levels={args.levels} problems={len(problems)} "
          f"samples={args.samples}")
    print(f"{len(tasks):,} rollouts, {len(done):,} already done, "
          f"{len(todo):,} to run -> {out_file}", flush=True)
    if args.dry_run or not todo:
        print("CELL_DONE", flush=True)
        return

    config = build_config(
        args.benchmark, args.model, args.method, teacher=args.teacher,
        tensor_parallel=args.tensor_parallel, joint_gpus=args.joint_gpus,
    )
    # The fleet worker's __call__ (bypassed here) puts the interpreter's bin
    # dir on PATH: vLLM shells out to venv binaries such as ninja for
    # torch.compile, and the joint decoder's engine subprocesses inherit
    # this environment.
    os.environ["PATH"] = (os.path.dirname(sys.executable) + os.pathsep
                          + os.environ.get("PATH", ""))
    WorkerCls = (JointDecodeInferenceWorker if args.method in MIXING
                 else VLLMInferenceWorker)
    # Only setup/process_batch are used; the worker's HTTP loop, which the
    # distributed runner uses, is never entered, so the address is unused.
    worker = WorkerCls("http://localhost:0")
    t0 = time.time()
    worker.setup(config)
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)

    try:
        _run_batches(worker, config, todo, benchmark, answers, out_file,
                     args.batch_size, t0)
    finally:
        # Exits the joint decoder's context, which stops its two engine
        # subprocesses; a no-op for the single-engine worker.
        worker.teardown()
    print("CELL_DONE", flush=True)


def _run_batches(worker, config, todo, benchmark, answers, out_file,
                 batch_size, t0) -> None:
    written = errors = 0
    with out_file.open("a") as sink:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            results = worker.process_batch(chunk, config)
            by_key = {t.task_key: t for t in chunk}
            for res in results:
                # Ungraded: grading is a separate pass over these files, so
                # one place decides what counts as a correct answer.
                row = result_row(res, by_key[res.task_key], benchmark,
                                 answers, grade_inline=False)
                sink.write(json.dumps(row) + "\n")
                errors += bool(res.is_error)
            sink.flush()
            written += len(results)
            print(f"{written:,}/{len(todo):,} "
                  f"({100 * written / len(todo):.0f}%) "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"wrote {written:,} rows"
          + (f", {errors:,} errors" if errors else "")
          + f" -> {out_file.resolve()}", flush=True)


if __name__ == "__main__":
    main()
