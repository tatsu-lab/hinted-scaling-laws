"""The answer-mixing control (appendix: when can hinting inform a forecast)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HEADLINE, HERE, draw_forecast_rows, set_style  # noqa: E402
from plot import pair_data  # noqa: E402


def synth_answer_mix(no_hint: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        d = no_hint.copy(); d["hint"] = float(h); d["acc"] = h + (1 - h) * no_hint["acc"].to_numpy(); rows.append(d)
    return pd.concat(rows, ignore_index=True)


def answermix_figure(out: Path) -> None:
    arms = {}
    for bench, fam, *_ in HEADLINE:
        no_hint, _ = pair_data(bench, fam)
        arms[(bench, fam)] = (no_hint, synth_answer_mix(no_hint))
    draw_forecast_rows("answer mixing", HEADLINE, arms, "Hint level $H$", out, slice_hint_fit="linear", slice_cap_fit="free",
                       ae_panel=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_style()
    answermix_figure(out / "answer_mixing.pdf")


if __name__ == "__main__":
    main()
