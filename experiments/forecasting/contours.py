"""Empirical iso-accuracy contours for every (benchmark, family) pair."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.tri as mtri

from common import HERE, model_ticks, set_style  # noqa: E402
from plot import BENCH_NAME, CAP_LABEL, INK, LABELS, METHODS, PAIRS, pair_data, size_tag  # noqa: E402

CONTOUR_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def heat_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap("gist_heat_r")
    return LinearSegmentedColormap.from_list("heat_trim", base(np.linspace(0.04, 1.0, 256)))


def draw_contours(ax, df, c0, names, cm):
    x, y, z = df["capability"].to_numpy(), df["hint"].to_numpy(), df["acc"].to_numpy()
    keep = df.groupby("capability")["hint"].transform("size").to_numpy() >= 3
    tri = mtri.Triangulation(x[keep], y[keep])
    cs = ax.tricontour(tri, z[keep], levels=CONTOUR_LEVELS, colors=[cm(0.2 + 0.8 * v) for v in CONTOUR_LEVELS], linewidths=0.8, zorder=2)
    ax.clabel(cs, fmt=lambda v: f"{100 * v:.0f}", fontsize=5, inline=True, inline_spacing=2, colors=INK)
    sc = ax.scatter(x, y, c=z, cmap=cm, vmin=0, vmax=1, s=8, edgecolors="none", zorder=3)
    model_ticks(ax, c0, names)
    ax.set_xlim(c0.min() - 0.3, c0.max() + 0.3); ax.set_ylim(-0.06, 1.06); ax.grid(False)
    return sc


def contours_figure(out: Path) -> None:
    pairs = PAIRS; cm = heat_cmap()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6.0, 1.9 * len(pairs) + 0.6), squeeze=False, layout="constrained")
    for r, (bench, fam, flab, _c) in enumerate(pairs):
        no_hint, full = pair_data(bench, fam)
        c0 = no_hint["capability"].to_numpy(); names = [size_tag(m) for m in no_hint["model"]]
        for j, meth in enumerate(METHODS):
            ax = axes[r][j]
            sc = draw_contours(ax, full[meth], c0, names, cm)
            if r == 0:
                ax.set_title(LABELS[meth])
            if r == len(pairs) - 1:
                ax.set_xlabel(CAP_LABEL[no_hint["capability_kind"].iloc[0]])
        axes[r][0].set_ylabel(f"{BENCH_NAME[bench]} {flab}\nHint level $H$", fontweight="bold")
    fig.colorbar(sc, ax=[row[-1] for row in axes], label="Accuracy", fraction=0.05, pad=0.02, shrink=0.5)
    fig.savefig(out); plt.close(fig); print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_style()
    contours_figure(out / "contours.pdf")


if __name__ == "__main__":
    main()
