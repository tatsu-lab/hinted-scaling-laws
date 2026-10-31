"""Draw ranking.pdf and ranking_lambda.pdf from experiments/ranking/out/rows__<bench>__<method>.json.

ranking.pdf, per method: accuracy against hint level, Kendall tau from one rollout per problem
at each level, and tau against the budget for every estimator. ranking_lambda.pdf: the
hint-prior ranking over the lambda sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranking import read_rollouts  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "results"
BENCHES = [("aime", "AIME", "Blues"), ("hle", "HLE", "Oranges")]
METHOD_LABEL = {"truncate": "Truncation", "mask": "Masking"}
HINT_PRIOR_COLOR = "#9467bd"
LAMBDAS = (1, 2, 4, 8, 16, 32, 64, 128)


def set_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold", "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3e0", "grid.linewidth": 0.5, "axes.axisbelow": True})


# --- data -------------------------------------------------------------
def load_rows(rows_dir, bench, method):
    fp = rows_dir / f"rows__{bench}__{method}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def series(rows, name):
    r = sorted((x for x in rows if x["method"] == name), key=lambda x: x["budget"])
    return np.array([x["budget"] for x in r]), np.array([x["mean_kendall_tau"] for x in r])


def level_of(name):
    return float(name.removeprefix("hinted_h"))


def tau_by_level(rows, k=1):
    out = {}
    for r in rows:
        if abs(float(r["budget"]) - k) > 1e-9:
            continue
        if r["method"] == "avg_at_k":
            out[0.0] = (r["mean_kendall_tau"], r["std_kendall_tau"])
        elif r["method"].startswith("hinted_h"):
            out[level_of(r["method"])] = (r["mean_kendall_tau"], r["std_kendall_tau"])
    return out


def hint_curves(data_dir, bench, method, models, cap):
    h0 = read_rollouts(data_dir / "rollouts" / f"{bench}__no_hint.csv")
    hinted = read_rollouts(data_dir / "rollouts" / f"{bench}__{method}.csv")
    hinted = hinted[hinted.sample_id < cap]
    out = {}
    for m in models:
        xs, ys = [0.0], [float(h0[h0.model == m].groupby("problem_id").passed.mean().mean())]
        g = hinted[hinted.model == m].groupby(["hint", "problem_id"]).passed.mean().groupby("hint").mean()
        xs += [round(float(h), 1) for h in g.index]; ys += [float(a) for a in g.values]
        out[m] = (xs, ys)
    return out


def short_label(model):
    name = model
    for suffix in ("-Instruct-v0.3", "-Instruct", "-it", "-chat-hf", "-Chat"):
        name = name.removesuffix(suffix)
    family, _, size = name.rpartition("-")
    return f"{family[:1].upper() + family[1:]} {size.upper()}"


# --- figures -----------------------------------------------------------------------------------
def ranking_figure(rows_dir, data_dir, methods, out):
    cfg = json.loads((data_dir / "ranking_config.json").read_text())
    n_rows, n_cols = len(methods), len(BENCHES)
    fig, axes = plt.subplots(n_rows, 3 * n_cols, figsize=(10.5, 1.9 * n_rows + 1.2), squeeze=False, layout="constrained")
    viridis = plt.get_cmap("viridis")
    model_handles = []
    for r, method in enumerate(methods):
        for c, (bench, title, cmap_name) in enumerate(BENCHES):
            res = load_rows(rows_dir, bench, method)
            rows = res["rows"]
            left, middle, right = axes[r][c], axes[r][n_cols + c], axes[r][2 * n_cols + c]
            # left: accuracy against hint level, shades light -> dark by unhinted accuracy
            curves = hint_curves(data_dir, bench, method, res["models"], cfg["benchmarks"][bench]["hinted_per_problem"])
            order = sorted(curves, key=lambda m: curves[m][1][0])
            shades = np.linspace(0.35, 0.95, len(order))
            for m, s in zip(order, shades):
                xs, ys = curves[m]
                line, = left.plot(xs, np.array(ys) * 100, color=plt.get_cmap(cmap_name)(s), marker="o", ms=2, lw=1.1)
                if r == 0:
                    model_handles.append((line, short_label(m)))
            left.set(ylim=(-3, 103), xticks=[0, 0.5, 1])
            # middle: tau of a one-rollout ranking at each level, +/- 1 s.d.
            by_h = tau_by_level(rows); hs = sorted(by_h)
            mean = np.array([by_h[h][0] for h in hs]); std = np.array([by_h[h][1] for h in hs])
            color = plt.get_cmap(cmap_name)(0.72)
            middle.fill_between(hs, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            middle.plot(hs, mean, color=color, marker="o", ms=2.3, lw=1.5)
            middle.axhline(0, color="gray", ls=":", lw=0.8)
            middle.set(xticks=[0, 0.5, 1])
            # right: tau against budget
            for name in sorted({x["method"] for x in rows if x["method"].startswith("hinted_h")}, key=level_of):
                right.plot(*series(rows, name), color=viridis(level_of(name)), lw=0.9, alpha=0.35)
            right.plot(*series(rows, "avg_at_k"), color="black", marker="o", ms=2.3, lw=1.5)
            right.plot(*series(rows, "hint_prior"), color=HINT_PRIOR_COLOR, marker="o", ms=2.3, lw=1.5)
            right.axhline(res["logloss_hlines"][0][2], color="firebrick", ls=(0, (4, 2)), lw=1.1)
            right.axhline(0, color="gray", ls=":", lw=0.8)
            right.set_xscale("log", base=2)
            for ax in (left, middle, right):
                if r == 0:
                    ax.set_title(title)
                if c > 0:
                    ax.tick_params(labelleft=False)
            if c == 0:
                left.set_ylabel(f"{METHOD_LABEL[method]}\nAccuracy (%)", fontweight="bold")
                middle.set_ylabel(r"Kendall's $\tau$"); right.set_ylabel(r"Kendall's $\tau$")
            if r == n_rows - 1:   # one x-axis name per block, on its first column
                if c == 0:
                    left.set_xlabel("Hint level $H$"); middle.set_xlabel("Hint level $H$"); right.set_xlabel("Rollouts per problem $k$")
                else:
                    left.set_xlabel(" "); middle.set_xlabel(" "); right.set_xlabel(" ")
    # the middle and right blocks share y-limits
    tau_axes = [ax for row in axes for ax in row[n_cols:]]
    lo = min(ax.get_ylim()[0] for ax in tau_axes); hi = max(ax.get_ylim()[1] for ax in tau_axes)
    for ax in tau_axes:
        ax.set_ylim(lo, hi)
    # legends: models under the left block, the rest under the right block; one colorbar for H
    fig.legend([h for h, _ in model_handles], [l for _, l in model_handles], loc="outside lower left", ncols=3, fontsize=5.5)
    fig.legend(handles=[
        Line2D([], [], color="black", marker="o", ms=2.3, lw=1.5, label="avg@$k$"),
        Line2D([], [], color=HINT_PRIOR_COLOR, marker="o", ms=2.3, lw=1.5, label="hint-prior ranking"),
        Line2D([], [], color="firebrick", ls=(0, (4, 2)), lw=1.1, label="log loss"),
        Line2D([], [], color="gray", ls=":", lw=0.8, label="chance"),
        Patch(facecolor="gray", alpha=0.3, label=r"$\pm 1$ s.d., one rollout per problem"),
    ], loc="outside lower right", ncols=3, fontsize=5.5)
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(0.1, 1.0)); sm.set_array([])
    fig.colorbar(sm, ax=[row[-1] for row in axes], label="Hint level $H$", fraction=0.05, pad=0.02, shrink=0.6)
    fig.savefig(out); plt.close(fig); print("wrote", out)


def lambda_figure(rows_dir, methods, out):
    n_rows, n_cols = len(methods), len(BENCHES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 1.8 * n_rows + 1.0), squeeze=False, sharex=True, sharey=True, layout="constrained")
    ramp = plt.get_cmap("plasma")
    for r, method in enumerate(methods):
        for c, (bench, title, _c) in enumerate(BENCHES):
            ax, rows = axes[r][c], load_rows(rows_dir, bench, method)["rows"]
            for i, lam in enumerate(LAMBDAS):
                name = "hint_prior" if lam == 8 else f"hint_prior_lambda{lam}"
                ax.plot(*series(rows, name), color=ramp(0.1 + 0.8 * i / (len(LAMBDAS) - 1)), lw=1.3, marker="o", ms=1.8,
                        label=rf"$\lambda={lam}$" + (" (used)" if lam == 8 else "") if (r, c) == (0, 0) else None)
            ax.plot(*series(rows, "avg_at_k"), color="black", lw=1.5, marker="o", ms=2, label="avg@$k$" if (r, c) == (0, 0) else None)
            ax.axhline(0, color="gray", ls=":", lw=0.7)
            ax.set_xscale("log", base=2)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"{METHOD_LABEL[method]}\nKendall's " + r"$\tau$", fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Completions per problem $k$")
    fig.legend(loc="outside lower center", ncols=3, fontsize=6)
    fig.savefig(out); plt.close(fig); print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-dir", type=Path, default=HERE / "out")
    ap.add_argument("--out-dir", type=Path, default=HERE, help="where the figures go (default: next to this script)")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--methods", nargs="+", default=["truncate", "mask"], choices=list(METHOD_LABEL))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    ranking_figure(a.rows_dir, a.data_dir, a.methods, a.out_dir / "ranking.pdf")
    lambda_figure(a.rows_dir, a.methods, a.out_dir / "ranking_lambda.pdf")


if __name__ == "__main__":
    main()
