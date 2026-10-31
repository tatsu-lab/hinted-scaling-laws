"""Convert graded runner output into tables in the format of `data/results/rollouts/`.

Reads `<out_root>/<benchmark>__<family>/<method>/<model>.jsonl` and writes
`<dest>/<benchmark>__<method>.csv`. Delphi checkpoints become a model and a `seed` column
(delphi-1e21-seed42 -> delphi_1e21, seed 42), and rollouts are ordered as in the released
tables.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

X_POOL = {"aime": 128, "hle": 64}


def read_runner_dir(d: Path) -> pd.DataFrame:
    parts = []
    for fp in sorted(d.glob("*.jsonl")):
        df = pd.read_json(fp, lines=True)
        if df.empty:
            continue
        # no verdict (judge failure) and parse_error (no extractable answer) both count as wrong
        parts.append(pd.DataFrame({"model": df["model"], "hint": df["hint_level"].round(4),
                                   "problem_id": df["problem_id"].astype(str), "sample_id": df["sample_id"].astype(str),
                                   "passed": (df["grade"] == "correct").astype(int)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["model", "hint", "problem_id", "sample_id", "passed"])


def rank(df: pd.DataFrame) -> pd.DataFrame:
    sid = df["sample_id"].astype(str)
    df = df.assign(_len=sid.str.len(), _sid=sid).sort_values(["_len", "_sid"], kind="stable")
    df["sample_id"] = df.groupby(["model", "hint", "problem_id"], sort=False).cumcount()
    return df.drop(columns=["_len", "_sid"])


def widen(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = [c for c in cols if c not in ("passed", "pool")]
    g = df.sort_values(keys + ["sample_id"], kind="stable").groupby(keys, sort=True)
    out = g["passed"].agg(lambda v: "".join(str(int(x)) for x in v)).reset_index()
    if "pool" in cols:
        out["pool"] = g["pool"].agg("".join).to_numpy()
    return out[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    tables: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for pair_dir in sorted(a.out_root.glob("*__*")):
        bench = pair_dir.name.split("__")[0]
        for method_dir in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
            df = read_runner_dir(method_dir)
            if len(df):
                tables[(bench, method_dir.name)].append(df)
    for (bench, method), parts in sorted(tables.items()):
        df = rank(pd.concat(parts, ignore_index=True))
        if method == "no_hint":
            df = df[df["hint"] == 0.0].copy()
            df["pool"] = (df["sample_id"] < X_POOL.get(bench, 128)).map({True: "x", False: "y"})
            cols = ["model", "problem_id", "passed", "pool"]
        else:
            df = df[df["hint"] > 0.0].copy()
            df["hint"] = df["hint"].map(lambda h: f"{h:g}")
            cols = ["model", "hint", "problem_id", "passed"]
        if bench == "gsm8k":
            ckpt = df["model"].str.extract(r"^delphi-([^-]+)(?:-seed(\d+))?$")
            df["model"] = "delphi_" + ckpt[0]
            df["seed"] = ckpt[1].fillna("original")
            cols = ["model", "seed"] + [c for c in cols if c not in ("model", "pool")]
        fp = a.dest / f"{bench}__{method}.csv"
        widen(df, cols).to_csv(fp, index=False)
        print(f"wrote {fp.name}: {len(df):,} rollouts, {df.model.nunique()} models")


if __name__ == "__main__":
    main()
