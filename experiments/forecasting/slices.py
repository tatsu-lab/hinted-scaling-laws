"""Figure 2 for every (benchmark, family) pair: each hint and capability slice fitted alone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, VIRIDIS, draw_cap_slices, draw_hint_slices, pct, set_style  # noqa: E402
from plot import BENCH_NAME, CAP_LABEL, LABELS, METHODS, PAIRS, pair_data, size_tag  # noqa: E402


def slices_figure(out: Path) -> None:
    pairs = PAIRS
    fig, axes = plt.subplots(len(pairs), 4, figsize=(10.5, 1.75 * len(pairs) + 0.8), squeeze=False, layout="constrained")
    for r, (bench, fam, flab, cmap_name) in enumerate(pairs):
        no_hint, full = pair_data(bench, fam)
        c0 = no_hint["capability"].to_numpy(); names = [size_tag(m) for m in no_hint["model"]]
        cmap = plt.get_cmap(cmap_name)
        for j, meth in enumerate(METHODS):
            df = full[meth]
            ah, ac = axes[r][2 * j], axes[r][2 * j + 1]
            draw_hint_slices(ah, df, c0, cmap); draw_cap_slices(ac, df, c0, names)
            for ax in (ah, ac):
                pct(ax)
            if r == 0:
                ah.set_title(f"{LABELS[meth]}: hint slices"); ac.set_title(f"{LABELS[meth]}: capability slices")
            if r == len(pairs) - 1:
                ah.set_xlabel("Hint level $H$"); ac.set_xlabel(CAP_LABEL[no_hint["capability_kind"].iloc[0]])
        axes[r][0].set_ylabel(f"{BENCH_NAME[bench]} {flab}\nAccuracy (%)", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=VIRIDIS, norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=[row[-1] for row in axes], label="Hint level $H$", fraction=0.04, pad=0.02, shrink=0.5)
    fig.savefig(out); plt.close(fig); print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_style()
    slices_figure(out / "slices.pdf")


if __name__ == "__main__":
    main()
