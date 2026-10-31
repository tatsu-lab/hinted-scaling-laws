"""Write the prompts the paper's hinted benchmarks send to a model, without running one.

Built by the runner's own prompt code (`hinted.runner.get_cell_prompts`), so each row matches
what `single_runner.py` would send.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hinted.benchmarks import HintType  # noqa: E402
from hinted.benchmarks.aime import AIMEBenchmark  # noqa: E402
from hinted.benchmarks.gsm8k_delphi import GSM8KDelphiBenchmark  # noqa: E402
from hinted.benchmarks.hle import HLESubsetBenchmark  # noqa: E402
from hinted.runner import get_cell_prompts, parse_task_key  # noqa: E402

METHODS = ["no_hint", "truncate", "mask", "logit_mix", "logit_mix_std"]
MIXING = ("logit_mix", "logit_mix_std")
HINT_TYPE = {"truncate": HintType.TRUNCATE, "mask": HintType.MASK,
             "logit_mix": HintType.LOGIT_MIXING, "logit_mix_std": HintType.LOGIT_MIXING,
             "no_hint": HintType.TRUNCATE}
DEFAULTS = {  # same as launch_all.py
    "aime": dict(hint_file="data/benchmarks/aime_solution_hints.jsonl", samples=10),
    "hle": dict(hint_file="data/benchmarks/hle_solution_hints.jsonl", samples=10),
    "gsm8k": dict(hint_file="data/benchmarks/gsm8k_solution_hints.jsonl", samples=32),
}


def make_benchmark(benchmark: str, method: str):
    """Mirror of single_runner.make_benchmark: only GSM8K varies by method."""
    if benchmark == "aime":
        return AIMEBenchmark(str(REPO / "data/benchmarks/aime2025_2026.jsonl"))
    if benchmark == "hle":
        return HLESubsetBenchmark(str(REPO / "data/benchmarks/hle.jsonl"))
    kw = dict(data_path=str(REPO / "data/benchmarks/gsm8k_delphi.jsonl"))
    if method == "mask":
        return GSM8KDelphiBenchmark(redact=False, hint_block=True, degradation="mask", **kw)
    if method in MIXING:
        return GSM8KDelphiBenchmark(redact=False, teacher=True, **kw)
    return GSM8KDelphiBenchmark(redact=True, hint_block=True, degradation="truncate", **kw)


def rows(benchmark: str, method: str, levels: list[float], samples: int,
         hint_file: str, limit: int | None):
    bench = make_benchmark(benchmark, method)
    problems = bench.load_with_hint(str(REPO / hint_file), limit=limit)
    answers = {p.problem_id: p.answer for p in problems}
    n_hints = {p.problem_id: len(p.hint) for p in problems}
    # get_cell_prompts reads only .name from the model config
    model = SimpleNamespace(name="prompt-only")
    for level in levels:
        tasks = get_cell_prompts(model, bench, level, HINT_TYPE[method],
                                 samples_per_problem=samples, problems=problems)
        for t in tasks:
            k = parse_task_key(t.task_key)
            row = {"benchmark": benchmark, "method": method, "hint": level,
                   "problem_id": k.problem_id, "sample_id": k.sample_id,
                   "hint_rollout": (None if level == 0 else k.sample_id % n_hints[k.problem_id]),
                   "answer": answers[k.problem_id], "prompt": t.prompt}
            if t.teacher_prompt is not None:
                row["teacher_prompt"] = t.teacher_prompt
            yield row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="hint levels; default 0.1..1.0 (no_hint: 0 only); for logit "
                         "mixing the teacher weights, any value >= 0")
    ap.add_argument("--samples", type=int, default=None,
                    help="samples per problem (default: the paper's, 10 or 32 for GSM8K)")
    ap.add_argument("--hint-file", default=None, help="default: the benchmark's file in data/benchmarks/")
    ap.add_argument("--limit", type=int, default=None, help="first N problems only")
    ap.add_argument("--out", type=Path, default=None, help="JSONL path (default: stdout)")
    ap.add_argument("--show", action="store_true", help="print the first prompt and exit")
    a = ap.parse_args()

    d = DEFAULTS[a.benchmark]
    if a.method == "no_hint":
        levels = [0.0]
    else:
        levels = a.levels or [round(0.1 * i, 1) for i in range(1, 11)]
    if any(l < 0 for l in levels) or (a.method not in MIXING and any(l > 1 for l in levels)):
        raise SystemExit("levels must lie in [0, 1] (logit mixing: >= 0)")
    it = rows(a.benchmark, a.method, levels, a.samples or d["samples"],
              a.hint_file or d["hint_file"], a.limit)

    if a.show:
        r = next(it)
        for key in ("benchmark", "method", "hint", "problem_id", "sample_id", "hint_rollout", "answer"):
            print(f"{key}: {r[key]}")
        print("\n--- prompt ---\n" + r["prompt"])
        if "teacher_prompt" in r:
            print("\n--- teacher_prompt ---\n" + r["teacher_prompt"])
        return

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
    sink = a.out.open("w") if a.out else sys.stdout
    n = 0
    for r in it:
        sink.write(json.dumps(r) + "\n")
        n += 1
    if a.out:
        sink.close()
        print(f"wrote {n} prompts -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
