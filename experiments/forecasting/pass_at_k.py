"""pass@k as a hinting method (appendix)."""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.special import gammaln

from common import HEADLINE, HERE, draw_forecast_rows, set_style  # noqa: E402
from plot import mix_pair_data, models_table, pair_data, read_rollouts, rollout_table  # noqa: E402


def pass_at_k(c: np.ndarray, n: np.ndarray, k: int) -> float:
    """Unbiased pass@k over problems, 1 - C(n-c, k) / C(n, k), with log-gamma."""
    ok = n >= k; c, n = c[ok], n[ok]
    fail = np.where(n - c >= k, np.exp(gammaln(n - c + 1) - gammaln(n - c - k + 1) - gammaln(n + 1) + gammaln(n - k + 1)), 0.0)
    return float((1 - fail).mean())


PASS_K_MAX = {"gsm8k": 64, "aime": 256, "hle": 256}   # the paper's depths: 64 per Delphi seed, 256 on AIME/HLE


@lru_cache(maxsize=None)
def unhinted_rollouts(bench: str) -> pd.DataFrame:
    """The benchmark's unhinted table, one row per rollout, read once for all its families."""
    return read_rollouts(rollout_table(bench, "no_hint"))


def pass_at_k_arm(bench: str, fam: str, no_hint: pd.DataFrame) -> pd.DataFrame:
    """pass@k per model from the unhinted pool as a hint ladder, H = log2 k, k = 2 .. PASS_K_MAX
    (per seed where the table has seeds, then the seed mean), plus the H = 0 anchor."""
    k_max = PASS_K_MAX[bench]
    df = unhinted_rollouts(bench)
    if "seed" not in df.columns:
        df = df.assign(seed="")
    meta = models_table(); meta = meta[(meta.benchmark == bench) & (meta.family == fam) & meta.in_forecasts]
    df = df[df.model.isin(meta.model)]
    per = df.groupby(["model", "seed", "problem_id"]).passed.agg(c="sum", n="size").reset_index()
    rows = []
    for (m, seed), g in per.groupby(["model", "seed"]):
        c, n = g["c"].to_numpy(), g["n"].to_numpy()
        k = 2
        while k <= min(k_max, int(n.min())):
            rows.append(dict(model=m, hint=float(np.log2(k)), acc=pass_at_k(c, n, k))); k *= 2
    pk = pd.DataFrame(rows).groupby(["model", "hint"], as_index=False).acc.mean()
    pk = pk.merge(meta[["model", "capability", "capability_kind"]], on="model")
    anchor = no_hint[["model", "capability", "capability_kind", "acc"]].assign(hint=0.0)
    return pd.concat([pk, anchor], ignore_index=True)


# the paper's colours (main.tex): pass@k blue (pkcol), hinting ensemble red (meancol), logit mixing purple
C_PK, C_ENSEMBLE, C_MIX = "#3b7dd8", "#d1495b", "#7b3fa0"


def passk_train_sizes(benchmark: str, n_models: int) -> list[int]:
    """The paper's pass@k figure: every train size from 2 on the small families, the five
    largest train sets (t = 7 .. 11 on Delphi's 12 rungs) otherwise."""
    return list(range(2 if n_models <= 6 else max(3, n_models - 5), n_models))


def passk_figure(out: Path) -> None:
    """pass@k forecasts against the paper's comparisons: the unhinted baseline, the hinting
    ensemble (mean of the masking and truncation floored-law forecasts) on every pair, and
    logit mixing on Delphi, all refitted on the same t smallest models."""
    arms, extras = {}, {}
    for bench, fam, *_ in HEADLINE:
        no_hint, full = pair_data(bench, fam)
        arms[(bench, fam)] = (no_hint, pass_at_k_arm(bench, fam, no_hint))
        extras[(bench, fam)] = [("hinting ensemble (masking and truncation)", C_ENSEMBLE, no_hint, {m: full[m] for m in ("mask", "truncate")})]
        if (bench, fam) == ("gsm8k", "delphi"):
            nh_mix, cells = mix_pair_data(bench, fam, "logit_mix")
            extras[(bench, fam)].append(("logit mixing (Delphi)", C_MIX, nh_mix, {"logit_mix": cells}))
        print(f"  {bench}/{fam}: pass@k to k = {2 ** int(arms[(bench, fam)][1]['hint'].max())}")
    # the paper draws every pair's pass@k hint slices in blue
    pairs = [(bench, fam, flab, "Blues") for bench, fam, flab, _ in HEADLINE]
    draw_forecast_rows("pass@k as a hinting method", pairs, arms, "$H = \\log_2 k$", out, slice_hint_fit=None, slice_cap_fit=None,
                       slices="joint", extras=extras, colour=C_PK,
                       train_sizes_fn=passk_train_sizes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_style()
    passk_figure(out / "pass_at_k.pdf")


if __name__ == "__main__":
    main()
