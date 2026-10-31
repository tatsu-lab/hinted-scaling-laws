"""The grid of cells the paper runs, and how to launch them.

Models, rollouts, hint files and levels are declared here and passed to `single_runner.py`, so
each emitted command fully specifies one cell. Filter with --benchmark, --family, --method and
--max-params.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = "experiments/query_hinted_benchmarks/single_runner.py"

# Hint levels for the hinting methods. H = 0 is not among them: there all
# of them reduce to the plain question (truncation shows nothing, masking
# would be all [MASK] so the hint is dropped, logit mixing has zero teacher
# weight), so the unhinted measurement is made once per pair as the
# `no_hint` method and shared as every method's H = 0 anchor — the
# convention the released tables follow. Logit mixing's H is the teacher's
# weight, which is unbounded.
HINT_LEVELS = [round(0.1 * i, 1) for i in range(1, 11)]
MIX_LEVELS_DELPHI = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                     0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4,
                     2.0, 4.0, 8.0, 16.0]
MIX_LEVELS_POSTTRAINED = [0.11, 0.25, 0.43, 0.67, 1.0, 1.5, 2.33, 4.0, 9.0, 16.0]

METHODS = ["no_hint", "truncate", "mask", "logit_mix", "logit_mix_std"]
MIXING = ("logit_mix", "logit_mix_std")

QWEN25 = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
          "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"]
QWEN3 = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
         "Qwen3-32B"]
QWEN35 = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B",
          "Qwen3.5-27B"]
GEMMA3 = ["gemma-3-270m-it", "gemma-3-1b-it", "gemma-3-4b-it",
          "gemma-3-12b-it", "gemma-3-27b-it"]
DELPHI_RUNGS = ["3e18", "9e18", "2e19", "3e19", "5e19", "9e19", "2e20", "3e20", "6e20",
                "1e21", "1e22"]
# three training seeds per rung (5e19 and 6e20 have no unsuffixed checkpoint:
# seed 0 stands in for it), the original 1e23 alone
DELPHI = [f"delphi-{r}{s}" for r in DELPHI_RUNGS
          for s in (("-seed0", "-seed42", "-seed62746") if r in ("5e19", "6e20")
                    else ("", "-seed42", "-seed62746"))] + ["delphi-1e23"]


@dataclass(frozen=True)
class Pair:
    """One (benchmark, family) ladder as the paper runs it.

    samples_per_problem is per method; logit mixing runs only when `teacher`
    is set, on `mix_models` at `mix_levels` under each rule in `mix_rules`."""

    benchmark: str
    family: str
    models: list[str]
    hint_file: str
    samples_per_problem: dict
    n_problems: int
    teacher: str | None = None
    mix_models: tuple = ()
    mix_levels: tuple = ()
    mix_rules: tuple = ()

    @property
    def key(self) -> str:
        return f"{self.benchmark}/{self.family}"

    def methods(self) -> list[str]:
        return [m for m in METHODS if m not in MIXING or m in self.mix_rules]

    def models_for(self, method: str) -> list[str]:
        return list(self.mix_models) if method in MIXING else self.models

    def levels_for(self, method: str) -> list[float]:
        if method == "no_hint":
            return [0.0]
        return list(self.mix_levels) if method in MIXING else HINT_LEVELS


_TEN = {m: 10 for m in METHODS}
_AIME = dict(benchmark="aime", hint_file="data/benchmarks/aime_solution_hints.jsonl",
             samples_per_problem=_TEN, n_problems=60)
_HLE = dict(benchmark="hle", hint_file="data/benchmarks/hle_solution_hints.jsonl",
            samples_per_problem=_TEN, n_problems=176)


def _aime_mixing(models: list[str]) -> dict:
    """The second-largest model teaches every model up to and including itself."""
    return dict(teacher=models[-2], mix_models=tuple(models[:-1]),
                mix_levels=tuple(MIX_LEVELS_POSTTRAINED), mix_rules=MIXING)


PAIRS: list[Pair] = [
    Pair(benchmark="gsm8k", family="delphi", models=DELPHI,
         hint_file="data/benchmarks/gsm8k_solution_hints.jsonl",
         samples_per_problem={"no_hint": 64, "truncate": 32, "mask": 32, "logit_mix": 32},
         n_problems=256, teacher="Llama-3.1-8B-Instruct", mix_models=tuple(DELPHI),
         mix_levels=tuple(MIX_LEVELS_DELPHI), mix_rules=("logit_mix",)),

    Pair(family="qwen25", models=QWEN25, **_AIME, **_aime_mixing(QWEN25)),
    Pair(family="qwen3", models=QWEN3, **_AIME, **_aime_mixing(QWEN3)),
    Pair(family="gemma3", models=GEMMA3, **_AIME, **_aime_mixing(GEMMA3)),

    Pair(family="qwen25", models=QWEN25, **_HLE),
    Pair(family="qwen3", models=QWEN3, **_HLE),
    Pair(family="gemma3", models=GEMMA3, **_HLE),
    Pair(family="qwen35", models=QWEN35, **_HLE),
]


def params_b(model: str) -> float:
    """Rough parameter count in billions, parsed from the model's name.

    Only used for --max-params, so approximate is fine. Two things it cannot
    see: the Delphi rungs are named by training FLOPs, so they are never
    filtered (delphi-1e23 is 25B, the largest model in the grid); and a
    mixture-of-experts name reports total rather than active parameters.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*([BbMm])", model)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val / 1000 if m.group(2).lower() == "m" else val


def cells(args) -> list[tuple[Pair, str, str]]:
    out = []
    for pair in PAIRS:
        if args.benchmark and pair.benchmark != args.benchmark:
            continue
        if args.family and pair.family != args.family:
            continue
        for method in pair.methods():
            if args.method and method != args.method:
                continue
            for model in pair.models_for(method):
                if args.max_params and params_b(model) > args.max_params:
                    continue
                out.append((pair, method, model))
    return out


def command(pair: Pair, method: str, model: str, args) -> str:
    """A complete specification of one cell, as a shell command."""
    levels = [f"{h:g}" for h in pair.levels_for(method)]
    cmd = [args.python, RUNNER,
           "--model", model,
           "--benchmark", pair.benchmark,
           "--method", method,
           "--samples", str(pair.samples_per_problem[method]),
           "--levels", *levels,
           "--out", f"{args.out_root}/{pair.benchmark}__{pair.family}/"
                    f"{method}/{model}.jsonl"]
    if method != "no_hint":
        cmd += ["--hint-file", pair.hint_file]
    if method in MIXING:
        cmd += ["--teacher", pair.teacher, "--joint-gpus", str(args.joint_gpus)]
    elif args.tensor_parallel != 1:
        cmd += ["--tensor-parallel", str(args.tensor_parallel)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return " ".join(cmd)


# Adapt to your scheduler. Logit mixing needs two GPUs; the single-engine
# methods get --tensor-parallel of them.
SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={logdir}/{job}.out
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["gsm8k", "aime", "hle"])
    ap.add_argument("--family")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--max-params", type=float, default=None,
                    help="skip models above this many billion parameters")
    ap.add_argument("--limit", type=int, default=None,
                    help="passed through: first N problems only")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--joint-gpus", type=int, default=2, choices=[1, 2])
    ap.add_argument("--python", default="python")
    ap.add_argument("--out-root", default="experiments/query_hinted_benchmarks/out")
    ap.add_argument("--print", dest="print_cmds", action="store_true",
                    help="print one shell command per cell")
    ap.add_argument("--sbatch", action="store_true",
                    help="write one sbatch script per cell under <out-root>/slurm")
    ap.add_argument("--submit", action="store_true",
                    help="with --sbatch, actually submit them")
    args = ap.parse_args()

    grid = cells(args)

    if args.print_cmds:
        for pair, method, model in grid:
            print(command(pair, method, model, args))
        return

    if args.sbatch:
        logdir = Path(args.out_root) / "slurm"
        logdir.mkdir(parents=True, exist_ok=True)
        for pair, method, model in grid:
            job = f"{pair.benchmark}-{pair.family}-{method}-{model}"
            script = logdir / f"{job}.sbatch"
            gpus = (args.joint_gpus if method in MIXING
                    else args.tensor_parallel)
            script.write_text(
                SBATCH_HEADER.format(job=job, logdir=logdir, gpus=gpus)
                + "\n" + command(pair, method, model, args) + "\n")
            if args.submit:
                subprocess.run(["sbatch", str(script)], check=True)
        print(f"{'submitted' if args.submit else 'wrote'} {len(grid)} jobs "
              f"under {logdir}")
        return

    # default: summarise the grid
    by_pair: dict = {}
    for pair, method, model in grid:
        by_pair.setdefault(pair.key, []).append((pair, method, model))
    print(f"{'pair':16s} {'models':>7s} {'cells':>6s} {'rollouts':>12s}")
    total_cells = total_rollouts = 0
    for key, cs in by_pair.items():
        pair = cs[0][0]
        rollouts = sum(len(pair.levels_for(m)) * pair.n_problems * pair.samples_per_problem[m]
                       for _p, m, _x in cs)
        total_cells += len(cs)
        total_rollouts += rollouts
        print(f"{key:16s} {len({m for _p, _me, m in cs}):7d} {len(cs):6d} "
              f"{rollouts:12,d}")
    print(f"{'TOTAL':16s} {'':7s} {total_cells:6d} {total_rollouts:12,d}")
    print("\n--print emits one command per cell; --sbatch writes job scripts.")


if __name__ == "__main__":
    main()
