"""Plot the paper's hinted scaling laws from `data/results/`.

`marginals` and `forecasts` cover the eight (benchmark, family) pairs under masking and
truncation, `mixing` the logit-mixing pairs. `--rollouts-dir` reads your own tables instead and
never falls back to the released ones.
"""

from __future__ import annotations

import argparse
import re
import statistics
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.optimize import least_squares
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "data" / "results"
ROLLOUTS = RESULTS / "rollouts"          # replaced by --rollouts-dir
OUT = HERE
# Rollouts per problem the fits use: the first 10 (by `sample_id`, the rank in the tables'
# canonical order) on AIME and HLE, all on GSM8K. The deeper h = 0 pools in
# `<bench>__no_hint` are for the ranking experiments (experiments/ranking) and pass@k.
FIT_ROLLOUTS = {"aime": 10, "hle": 10, "gsm8k": None}

METHODS = ["mask", "truncate"]
LABELS = {"mask": "Masking", "truncate": "Truncation", "logit_mix": "Logit mixing",
          "logit_mix_std": "Logit mixing, std rescaling"}

# (benchmark, family, display name, colormap for the model shades)
PAIRS = [("aime", "qwen25", "Qwen2.5", "Blues"),
         ("aime", "qwen3", "Qwen3", "Reds"),
         ("aime", "gemma3", "Gemma-3", "Greens"),
         ("hle", "qwen25", "Qwen2.5", "Blues"),
         ("hle", "qwen3", "Qwen3", "Reds"),
         ("hle", "gemma3", "Gemma-3", "Greens"),
         ("hle", "qwen35", "Qwen3.5", "Purples"),
         ("gsm8k", "delphi", "Delphi", "Oranges")]
# the pairs behind the paper's headline forecast numbers
HEADLINE_PAIRS = {"gsm8k/delphi", "aime/qwen25", "aime/gemma3", "hle/qwen3", "hle/qwen35"}
# logit mixing: (benchmark, family, display name, colormap, rule)
MIX_PAIRS = [("gsm8k", "delphi", "Delphi", "Oranges", "logit_mix"),
             ("aime", "qwen25", "Qwen2.5", "Blues", "logit_mix"),
             ("aime", "gemma3", "Gemma-3", "Greens", "logit_mix"),
             ("aime", "qwen3", "Qwen3", "Reds", "logit_mix"),
             ("aime", "qwen25", "Qwen2.5", "Blues", "logit_mix_std"),
             ("aime", "gemma3", "Gemma-3", "Greens", "logit_mix_std"),
             ("aime", "qwen3", "Qwen3", "Reds", "logit_mix_std")]
MIX_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0]    # capability-marginal slices
BENCH_NAME = {"gsm8k": "GSM8K", "aime": "AIME", "hle": "HLE"}
CAP_LABEL = {"log10_params_b": "Parameters (log scale)",
             "log10_train_flops": "Training FLOPs (log scale)"}
# y-axis ceiling of each forecast row, as in the paper
FORECAST_YMAX = {("gsm8k", "delphi"): 0.4, ("hle", "qwen3"): 0.4,
                 ("hle", "qwen35"): 0.6}

INK, INK_2 = "#16191d", "#5b646e"
C_BASE, C_MEAN, C_FAINT, C_WIN = "#0b0b0b", "#d1495b", "#a9a8a4", "#63a95e"
C_FIT, C_HELD = "#52514e", "#eda100"
DASHES = [(0, (4.5, 1.6)), (0, (1.3, 1.5))]
CHECK = MPath([(-1.0, 0.05), (-0.3, -0.75), (1.0, 0.8)],
              [MPath.MOVETO, MPath.LINETO, MPath.LINETO])
VIRIDIS = plt.get_cmap("viridis")


# ---------------------------------------------------------------------------
# Data and fits
# ---------------------------------------------------------------------------

def sigmoid(z):
    return expit(z)


_MODELS = None


def models_table() -> pd.DataFrame:
    """data/results/models.csv: one row per (benchmark, model) with family and capability."""
    global _MODELS
    if _MODELS is None:
        _MODELS = pd.read_csv(RESULTS / "models.csv")
    return _MODELS


def read_rollouts(fp: Path) -> pd.DataFrame:
    """Expand a release rollout table to one row per rollout.

    The tables hold one row per (model[, seed][, hint], problem) with `passed` a string of
    0/1 characters in rank order and, at H = 0, `pool` a parallel string of x/y/u
    characters. The position in the string is the rollout's rank, returned here as
    `sample_id`."""
    w = pd.read_csv(fp, dtype=str, keep_default_na=False)
    n = w["passed"].str.len().to_numpy()
    long = w.loc[w.index.repeat(n), [c for c in w.columns if c not in ("passed", "pool")]].reset_index(drop=True)
    long["sample_id"] = np.concatenate([np.arange(k) for k in n])
    long["passed"] = (np.frombuffer("".join(w["passed"]).encode(), dtype="S1") == b"1").astype(int)
    if "pool" in w.columns:
        long["pool"] = pd.Series(np.frombuffer("".join(w["pool"]).encode(), dtype="S1").astype(str)).map(
            {"x": "x", "y": "y", "u": "unused"}).to_numpy()
    if "hint" in long.columns:
        long["hint"] = long["hint"].astype(float)
    return long


def rollout_table(benchmark: str, method: str) -> Path:
    """`<benchmark>__<method>.csv` in ROLLOUTS; an error, not a fallback, when it is missing."""
    fp = ROLLOUTS / f"{benchmark}__{method}.csv"
    if not fp.exists():
        raise SystemExit(f"{fp} not found. Every table a pair needs, including "
                         f"{benchmark}__no_hint.csv, must be in the rollouts directory.")
    return fp


@lru_cache(maxsize=None)
def load_table(benchmark: str, method: str) -> pd.DataFrame:
    """One row per (model, hint): the accuracy over the first FIT_ROLLOUTS rollouts per
    problem, as k correct of n. Where the table has a `seed` column (Delphi), acc is the
    mean of the seeds' accuracies and k = acc * n."""
    df = read_rollouts(rollout_table(benchmark, method))
    thin = FIT_ROLLOUTS[benchmark]
    if thin is not None:
        df = df[df["sample_id"] < thin]
    if method == "no_hint":
        df = df.assign(hint=0.0)
    df["hint"] = df["hint"].astype(float).round(4)
    if "seed" not in df.columns:
        df = df.assign(seed="")
    g = df.groupby(["model", "hint", "seed"], as_index=False).agg(
        k=("passed", "sum"), n=("passed", "size"))
    g["acc"] = g["k"] / g["n"]
    cells = g.groupby(["model", "hint"], as_index=False).agg(acc=("acc", "mean"), n=("n", "sum"))
    cells["k"] = cells["acc"] * cells["n"]
    return cells


def load(benchmark: str, family: str, method: str) -> pd.DataFrame:
    """One row per (model, hint) of `family` from `rollouts/<benchmark>__<method>.csv`,
    joined with the model metadata."""
    cells = load_table(benchmark, method)
    meta = models_table()
    meta = meta[(meta["benchmark"] == benchmark) & (meta["family"] == family) & meta["in_forecasts"]]
    out = cells.merge(meta[["model", "capability", "capability_kind"]], on="model", how="inner")
    return out[["model", "capability", "capability_kind", "hint", "k", "n", "acc"]].reset_index(drop=True)


def pair_data(benchmark: str, family: str, methods=None):
    """The unhinted row, sorted by capability, and per method that method's
    hinted scores stacked with the shared H = 0 row."""
    no_hint = (load(benchmark, family, "no_hint")
               .sort_values("capability").reset_index(drop=True))
    full = {m: pd.concat([load(benchmark, family, m), no_hint.assign(hint=0.0)],
                         ignore_index=True) for m in (methods or METHODS)}
    return no_hint, full


def mix_pair_data(benchmark: str, family: str, rule: str):
    """Logit mixing's (no_hint, cells). no_hint keeps every model, the largest last, as
    the forecast target; the cells hold the students' (every model with mixing scores)
    hinted scores and their unhinted row, so a model enters the fit only as a student."""
    no_hint = (load(benchmark, family, "no_hint")
               .sort_values("capability").reset_index(drop=True))
    mix = load(benchmark, family, rule)
    students = set(mix["model"])
    no_hint = no_hint[no_hint["model"].isin(students) | (no_hint.index == len(no_hint) - 1)]
    cells = pd.concat([mix, no_hint[no_hint["model"].isin(students)].assign(hint=0.0)],
                      ignore_index=True)
    return no_hint.reset_index(drop=True), cells


# the capability axis is centred here before fitting, which only moves b0
C_SHIFT = 20.0


def law(b, c, h):
    """The floored sigmoid law, b = [b1, b2, b0, A, L, F]."""
    z = b[0] * (np.asarray(c, dtype=float) - C_SHIFT) + b[1] * np.asarray(h, dtype=float) + b[2]
    return np.maximum(b[5], b[4] + (b[3] - b[4]) * sigmoid(z))


_FITS: dict = {}


def fit_joint(df: pd.DataFrame) -> np.ndarray:
    """[b1, b2, b0, A, L, F] of the floored law, multi-start least squares on the
    clipped prediction (b1, b2 >= 0, A in (0, 1], L in [-50, 1), F in [0, 1))."""
    c, h = df["capability"].to_numpy(dtype=float), df["hint"].to_numpy(dtype=float)
    y = df["acc"].to_numpy(dtype=float)
    key = (c.tobytes(), h.tobytes(), y.tobytes())
    if key in _FITS:
        return _FITS[key]

    def resid(x):
        return law(x, c, h) - y

    best = None
    for b1 in (0.5, 2.0):
        for b2 in (0.5, 2.0, 8.0):
            for b0 in (-4.0, -2.0, 0.0):
                for A0, L0 in ((0.9, 0.0), (1.0, -0.1), (0.9, -0.5)):
                    for F0 in (0.0, 0.01, 0.02):
                        r = least_squares(resid, x0=[b1, b2, b0, A0, L0, F0],
                                          bounds=([0.0, 0.0, -np.inf, 1e-3, -50.0, 0.0],
                                                  [np.inf, np.inf, np.inf, 1.0, 0.999, 0.999]))
                        if best is None or r.cost < best.cost:
                            best = r
    _FITS[key] = best.x
    return best.x


def fit_capability(c: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[a1, a0] of acc = sigmoid(a1*C + a0), a1 >= 0 -- the baseline.

    Multi-start: on a floor-heavy ladder (most models near 0) this fit is
    ill-conditioned and a single start lands in a poor local optimum, which
    would flatter the hinted methods it is compared against.
    """
    def resid(b):
        return sigmoid(b[0] * c + b[1]) - y

    best = None
    for x0 in ([1.0, -10.0], [1.0, -c.mean()], [0.5, -0.5 * c.mean()],
               [3.0, -3.0 * c.mean()], [10.0, -10.0 * c.mean()]):
        r = least_squares(resid, x0=x0, bounds=([0.0, -np.inf], [np.inf, np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def forecast(no_hint: pd.DataFrame, full: dict, t: int) -> dict:
    """Fit on the t smallest models; predict H = 0 for the rest. Errors are
    absolute errors at the LARGEST model, which every train size holds out."""
    c0, y0 = no_hint["capability"].to_numpy(), no_hint["acc"].to_numpy()
    boundary = c0[t - 1]
    cgrid = np.linspace(c0.min() - 0.2, c0.max() + 0.35, 300)
    a = fit_capability(c0[:t], y0[:t])
    c_top, y_top = c0[-1], y0[-1]
    curves, preds, errs = [], [], {}
    for meth, df in full.items():
        b = fit_joint(df[df["capability"] <= boundary + 1e-9])
        curves.append(law(b, cgrid, 0.0))
        preds.append(float(law(b, c_top, 0.0)))
        errs[meth] = abs(preds[-1] - y_top)
    errs["baseline"] = float(abs(sigmoid(a[0] * c_top + a[1]) - y_top))
    errs["mean"] = float(abs(np.mean(preds) - y_top))
    return dict(c0=c0, y0=y0, boundary=boundary, cgrid=cgrid,
                base_curve=sigmoid(a[0] * cgrid + a[1]), curves=curves,
                mean_curve=np.mean(curves, axis=0), errs=errs)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def size_tag(model: str) -> str:
    """'Qwen2.5-0.5B-Instruct' -> '500M', 'gemma-3-27b-it' -> '27B',
    'delphi_2e19' -> '1.8e19' (a Delphi rung's recorded training budget, models.csv;
    the names round it)."""
    if model.startswith("delphi"):
        meta = models_table()
        cap = meta.loc[meta["model"] == model, "capability"].iloc[0]
        return f"{10 ** cap:.2g}".replace("e+", "e")
    m = re.findall(r"(\d+\.?\d*)\s*([BbMm])", model)
    if not m:
        return model
    val, unit = float(m[-1][0]), m[-1][1].upper()
    if unit == "B" and val < 1:
        return f"{val * 1000:.0f}M"
    return f"{val:g}{unit}"


def style(ax, ymax: float) -> None:
    ax.set_ylim(-0.03 * ymax, ymax * 1.03)
    ax.grid(color="#e4e3e0", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=5.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def pct_axis(ax, ymax: float) -> None:
    step = 0.2 if ymax > 0.45 else 0.1 if ymax > 0.15 else 0.05
    ticks = np.arange(0.0, ymax + 1e-9, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v * 100:.0f}" for v in ticks])


def model_ticks(ax, c0, names, label_ends: bool) -> None:
    """A tick at every model; the smallest and largest labelled."""
    ax.set_xticks(c0)
    labs = [""] * len(c0)
    if label_ends:
        labs[0], labs[-1] = size_tag(names[0]), size_tag(names[-1])
    ax.set_xticklabels(labs, fontsize=4.8)
    ax.tick_params(axis="x", length=2)


def split_plot(ax, grid, y, boundary, colour, lw, dash, alpha=1.0, z=3):
    """Solid over the fitted span, dashed over the extrapolated span."""
    lo, hi = grid <= boundary, grid >= boundary
    ax.plot(grid[lo], y[lo], color=colour, lw=lw, alpha=alpha, zorder=z)
    ax.plot(grid[hi], y[hi], color=colour, lw=lw, ls=dash, alpha=alpha, zorder=z)


def mae_stamp(ax, mae: float) -> None:
    ax.text(0.04, 0.96, f"MAE {mae:.3f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=4.8, color=INK, zorder=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))


def to_axis(h):
    """Logit mixing's hint axis: H / (1 + H), so 0 .. 16 fits on [0, 1)."""
    h = np.asarray(h, dtype=float)
    return h / (1.0 + h)


def hint_slices(ax, df, b, c0, names, cmap, shades, mixing: bool) -> list:
    """Accuracy against H, one curve per model; returns the legend handles."""
    hgrid = np.linspace(0, 16, 400) if mixing else np.linspace(0, 1, 200)
    xs = to_axis if mixing else (lambda h: h)
    handles = []
    for cap, name, sh in zip(c0, names, shades):
        sub = df[np.isclose(df["capability"], cap)].sort_values("hint")
        if sub.empty:
            continue
        ax.plot(xs(sub["hint"]), sub["acc"], "o", ms=3.2, color=cmap(sh),
                markeredgewidth=0, zorder=4)
        (line,) = ax.plot(xs(hgrid), law(b, cap, hgrid), color=cmap(sh), lw=1.0, zorder=3)
        handles.append((line, size_tag(name)))
    if mixing:
        ticks = [0, 0.5, 1, 2, 4, 16]
        ax.set_xticks(to_axis(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(-0.03, 1.0)
    else:
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
    return handles


def capability_slices(ax, df, b, cgrid, levels, colour) -> None:
    """Accuracy against C, one curve per hint level."""
    for h in levels:
        sub = df[np.isclose(df["hint"], h)].sort_values("capability")
        ax.plot(sub["capability"], sub["acc"], "o", ms=3.2, color=colour(h),
                markeredgewidth=0, zorder=4)
        ax.plot(cgrid, law(b, cgrid, h), color=colour(h), lw=1.0, zorder=3)


def benchmark_headings(fig, top_axes, fig_h) -> None:
    """Benchmark names over their group of columns."""
    for bench in BENCH_NAME:
        axes = [ax for bch, ax in top_axes if bch == bench]
        if not axes:
            continue
        x0 = axes[0].get_position().x0
        x1 = axes[-1].get_position().x1
        y = axes[0].get_position().y1 + 0.32 / fig_h
        fig.text((x0 + x1) / 2, y, BENCH_NAME[bench], ha="center", va="bottom",
                 color=INK, fontsize=8.5, fontweight="bold")
        fig.add_artist(plt.Line2D([x0, x1], [y - 0.05 / fig_h] * 2, color=INK_2,
                                  lw=0.6, transform=fig.transFigure))


def page(nc: int, rows: list[float]):
    """A figure with nc equal columns and rows of the given heights (inches)."""
    panel_w = 1.3
    left_in, right_in, top_in, bottom_in = 0.95, 0.15, 0.75, 0.15
    gap_w, gap_h = 0.26, 0.38
    fig_w = left_in + right_in + nc * panel_w + (nc - 1) * gap_w
    fig_h = top_in + bottom_in + sum(rows) + (len(rows) - 1) * gap_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        len(rows), nc, height_ratios=rows,
        left=left_in / fig_w, right=1 - right_in / fig_w,
        top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
        wspace=gap_w / panel_w, hspace=gap_h / (sum(rows) / len(rows)))
    return fig, gs, fig_w, fig_h


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

def plot_marginals(pairs: list, out: Path) -> None:
    """Every pair on one page, every panel the same size.

    One column per (benchmark, family). The top block is the hint marginals
    (accuracy against H, one curve per model), the bottom block the
    capability marginals (accuracy against C, one curve per hint level); in
    each block the rows are the methods. Every curve in a column and
    row comes from that pair and method's single fitted law.
    """
    nc, nm = len(pairs), len(METHODS)
    panel_h, legend_h = 0.95, 0.85
    heights = [panel_h] * nm + [legend_h] + [panel_h] * nm + [0.45]
    fig, gs, fig_w, fig_h = page(nc, heights)
    hint_row0, cap_row0 = 0, nm + 1
    top_axes = []

    for j, (bench, fam, fam_name, cmap_name) in enumerate(pairs):
        no_hint, full = pair_data(bench, fam)
        c0 = no_hint["capability"].to_numpy()
        names = no_hint["model"].tolist()
        cap_kind = no_hint["capability_kind"].iloc[0]
        cmap = plt.get_cmap(cmap_name)
        shades = np.linspace(0.35, 0.95, len(c0))
        cgrid = np.linspace(c0.min() - 0.15, c0.max() + 0.3, 300)
        ymax = min(1.0, max(0.2, max(float(d["acc"].max()) for d in full.values()) * 1.12))
        handles = []

        for r, meth in enumerate(METHODS):
            df = full[meth]
            b = fit_joint(df)
            mae = float(np.abs(law(b, df["capability"], df["hint"]) - df["acc"]).mean())

            ax = fig.add_subplot(gs[hint_row0 + r, j])
            hs = hint_slices(ax, df, b, c0, names, cmap, shades, mixing=False)
            if r == 0:
                handles = hs
            mae_stamp(ax, mae)
            style(ax, ymax)
            pct_axis(ax, ymax)
            if r == 0:
                ax.set_title(fam_name, color=INK, fontsize=7, fontweight="bold", pad=3)
                top_axes.append((bench, ax))
            if r == nm - 1:
                ax.set_xlabel("Hint level $H$", color=INK_2, fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{LABELS[meth]}\nAccuracy (%)", color=INK,
                              fontsize=6.5, fontweight="bold")

            axc = fig.add_subplot(gs[cap_row0 + r, j])
            capability_slices(axc, df, b, cgrid, sorted(df["hint"].unique()), VIRIDIS)
            mae_stamp(axc, mae)
            axc.set_xlim(cgrid[0], cgrid[-1])
            model_ticks(axc, c0, names, label_ends=True)
            style(axc, ymax)
            pct_axis(axc, ymax)
            if r == 0:
                axc.set_title(fam_name, color=INK, fontsize=7, fontweight="bold", pad=3)
            if r == nm - 1:
                axc.set_xlabel(CAP_LABEL.get(cap_kind, "Capability"), color=INK_2,
                               fontsize=6)
            if j == 0:
                axc.set_ylabel(f"{LABELS[meth]}\nAccuracy (%)", color=INK,
                               fontsize=6.5, fontweight="bold")

        # this family's models, under its hint-marginal column
        lax = fig.add_subplot(gs[nm, j])
        lax.axis("off")
        lax.legend([h for h, _ in handles], [l for _, l in handles], loc="upper center",
                   ncol=2, frameon=False, fontsize=5.3, handlelength=1.3,
                   columnspacing=0.8, labelspacing=0.25,
                   title=f"{fam_name} models", title_fontsize=5.6)

    benchmark_headings(fig, top_axes, fig_h)

    # block labels down the far left
    for row0, text in ((hint_row0, "Hint marginals"), (cap_row0, "Capability marginals")):
        y0 = gs[row0 + nm - 1, 0].get_position(fig).y0
        y1 = gs[row0, 0].get_position(fig).y1
        fig.text(0.18 / fig_w, (y0 + y1) / 2, text, rotation=90, ha="center",
                 va="center", color=INK, fontsize=9, fontweight="bold")

    level_ax = fig.add_subplot(gs[len(heights) - 1, :])
    level_ax.axis("off")
    levels = np.linspace(0, 1, 11)
    level_ax.legend([plt.Line2D([], [], color=VIRIDIS(h), lw=1.6) for h in levels],
                    [f"{h:.1f}" for h in levels], loc="center", ncol=11,
                    frameon=False, fontsize=6, handlelength=1.6,
                    title="Hint level $H$ (capability marginals)", title_fontsize=6.5)
    fig.suptitle("Hint and capability marginals of the joint law", color=INK,
                 fontsize=11, fontweight="bold", y=1 - 0.12 / fig_h)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

def train_sizes(benchmark: str, n_models: int) -> list[int]:
    return list(range(7 if benchmark == "gsm8k" else 2, n_models))


def draw_forecast(ax, fc: dict, t: int, names: list, ymax: float, first: bool,
                  faint: bool = True) -> None:
    """One forecast panel: the baseline, each method's H = 0 curve (faint) and their
    mean, the fitted (grey) and held-out (orange) models."""
    c0, y0, cgrid, boundary = fc["c0"], fc["y0"], fc["cgrid"], fc["boundary"]
    ax.axvspan(cgrid[0], boundary, color="#0b0b0b", alpha=0.05, zorder=0)
    split_plot(ax, cgrid, fc["base_curve"], boundary, C_BASE, 1.3, (0, (2, 2)), z=4)
    if faint:
        for i, cur in enumerate(fc["curves"]):
            split_plot(ax, cgrid, cur, boundary, C_FAINT, 0.8, DASHES[i], alpha=0.4, z=2)
    split_plot(ax, cgrid, fc["mean_curve"], boundary, C_MEAN, 1.3, (0, (2, 2)), z=4)
    if fc["errs"]["mean"] < fc["errs"]["baseline"]:
        ax.plot([0.11], [0.86], transform=ax.transAxes, marker=CHECK, ms=5.5,
                ls="", markeredgecolor=C_WIN, markerfacecolor="none",
                markeredgewidth=1.2, alpha=0.6, clip_on=False, zorder=7)
    ax.scatter(c0[:t], y0[:t], s=9, color=C_FIT, edgecolors="white",
               linewidths=0.3, zorder=6)
    ax.scatter(c0[t:], y0[t:], s=11, color=C_HELD, edgecolors="white",
               linewidths=0.3, zorder=6)
    ax.set_xlim(cgrid[0], cgrid[-1])
    model_ticks(ax, c0, names, label_ends=first)
    style(ax, ymax)
    pct_axis(ax, ymax)


def forecast_legend(fig, headline: str, methods: list) -> None:
    hnd = [plt.Line2D([], [], color=C_BASE, lw=1.4), plt.Line2D([], [], color=C_MEAN, lw=1.4)]
    lab = ["Baseline", headline]
    for i, meth in enumerate(methods):
        hnd.append(plt.Line2D([], [], color=C_FAINT, lw=1.1, alpha=0.75, ls=DASHES[i]))
        lab.append(LABELS[meth])
    hnd += [plt.Line2D([], [], marker="o", ls="", ms=3.4, color=C_FIT),
            plt.Line2D([], [], marker="o", ls="", ms=3.6, color=C_HELD),
            plt.Line2D([], [], marker=CHECK, ms=6, ls="", markeredgecolor=C_WIN,
                       markerfacecolor="none", markeredgewidth=1.2, alpha=0.6)]
    lab += ["Fitted models", "Held-out models", "Beats baseline"]
    fig.legend(hnd, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0, columnspacing=1.0,
               handletextpad=0.4)


def plot_forecasts(pairs: list, out: Path) -> list[dict]:
    data = {p[:2]: pair_data(p[0], p[1]) for p in pairs}
    tsets = {k: train_sizes(k[0], len(v[0])) for k, v in data.items()}
    ncol = max(len(v) for v in tsets.values())
    widths = [1.0] * ncol + [0.35, 1.0]
    fig = plt.figure(figsize=(1.45 * ncol + 2.4, 1.35 * len(pairs) + 1.1))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(pairs), len(widths), width_ratios=widths,
                          left=0.09, right=0.985, top=1 - 0.45 / fig.get_figheight(),
                          bottom=0.85 / fig.get_figheight(), wspace=0.14, hspace=0.6)
    rows = []

    for r, (bench, fam, fam_name, _cmap) in enumerate(pairs):
        no_hint, full = data[(bench, fam)]
        names = no_hint["model"].tolist()
        ymax = FORECAST_YMAX.get((bench, fam), 1.0)
        ts = tsets[(bench, fam)]
        errs = {k: [] for k in METHODS + ["baseline", "mean"]}
        for j, t in enumerate(ts):
            fc = forecast(no_hint, full, t)
            for k in errs:
                errs[k].append(fc["errs"][k])
            rows.append(dict(pair=f"{bench}/{fam}", t=t,
                             baseline=fc["errs"]["baseline"], ensemble=fc["errs"]["mean"]))

            ax = fig.add_subplot(gs[r, j])
            draw_forecast(ax, fc, t, names, ymax, first=(j == 0))
            ax.set_title(f"{t} fitted models", color=INK, fontsize=6, pad=3)
            if j:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(f"{BENCH_NAME[bench]}\n{fam_name}", color=INK,
                              fontsize=6.5, fontweight="bold")
        for j in range(len(ts), ncol):
            fig.add_subplot(gs[r, j]).axis("off")

        axm = fig.add_subplot(gs[r, len(widths) - 1])
        c0 = no_hint["capability"].to_numpy()
        fracs = [float(10 ** (c0[t - 1] - c0[-1])) for t in ts]
        for i, meth in enumerate(METHODS):
            axm.plot(fracs, errs[meth], lw=0.8, alpha=0.4, color=C_FAINT,
                     ls=DASHES[i], zorder=2)
        for key, colour in (("baseline", C_BASE), ("mean", C_MEAN)):
            axm.plot(fracs, errs[key], marker="o", ms=2.4, lw=1.1, color=colour, zorder=3)
        axm.set_xscale("log")
        axm.set_xticks(fracs)
        axm.set_xticklabels([f"{100 * f:.2g}%" for f in fracs], rotation=70,
                            ha="right", rotation_mode="anchor", fontsize=4.4)
        axm.minorticks_off()
        lo, hi = np.log10(min(fracs)), np.log10(max(fracs))
        pad = 0.12 * (hi - lo) if hi > lo else 0.3
        axm.set_xlim(10 ** (lo - pad), 10 ** (hi + pad))
        style(axm, max(max(v) for v in errs.values()) * 1.15)
        axm.set_ylabel("Abs. error", color=INK_2, fontsize=6, labelpad=1.0)
        if r == 0:
            axm.set_title("AE, largest model", color=INK, fontsize=6.5,
                          fontweight="bold", pad=3)
        if r == len(pairs) - 1:
            axm.set_xlabel("Fraction of\ntarget run size", color=INK_2, fontsize=6,
                           linespacing=1.0)

    forecast_legend(fig, "Hinting ensemble", METHODS)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return rows


def report(rows: list[dict]) -> None:
    print(f"\n{'pair':14s} {'t':>3s} {'baseline':>9s} {'ensemble':>9s}")
    for row in rows:
        win = "  beats baseline" if row["ensemble"] < row["baseline"] else ""
        print(f"{row['pair']:14s} {row['t']:3d} {row['baseline']:9.4f} "
              f"{row['ensemble']:9.4f}{win}")
    for label, sub in (("all pairs", rows),
                       ("the paper's headline pairs",
                        [r for r in rows if r["pair"] in HEADLINE_PAIRS])):
        if not sub:
            continue
        wins = sum(r["ensemble"] < r["baseline"] for r in sub)
        print(f"\n{label}: {len(sub)} panels; the ensemble beats the baseline in {wins}")
        print(f"median absolute error at the largest model: baseline "
              f"{100 * statistics.median(r['baseline'] for r in sub):.1f} points, "
              f"ensemble {100 * statistics.median(r['ensemble'] for r in sub):.1f}")


# ---------------------------------------------------------------------------
# Logit mixing
# ---------------------------------------------------------------------------

def plot_mixing_marginals(pairs: list, out: Path) -> None:
    """One column per (pair, rule): hint marginals over capability marginals."""
    heights = [0.95, 0.85, 0.95, 0.45]
    fig, gs, fig_w, fig_h = page(len(pairs), heights)
    top_axes = []
    level_colour = lambda h: VIRIDIS(float(to_axis(h)) / to_axis(16.0))
    for j, (bench, fam, fam_name, cmap_name, rule) in enumerate(pairs):
        no_hint, df = mix_pair_data(bench, fam, rule)
        no_hint = no_hint[no_hint["model"].isin(df["model"])]
        c0 = no_hint["capability"].to_numpy()
        names = no_hint["model"].tolist()
        cmap = plt.get_cmap(cmap_name)
        shades = np.linspace(0.35, 0.95, len(c0))
        cgrid = np.linspace(c0.min() - 0.15, c0.max() + 0.3, 300)
        ymax = min(1.0, max(0.2, float(df["acc"].max()) * 1.12))
        b = fit_joint(df)
        mae = float(np.abs(law(b, df["capability"], df["hint"]) - df["acc"]).mean())
        title = fam_name + (" (std)" if rule == "logit_mix_std" else "")

        ax = fig.add_subplot(gs[0, j])
        handles = hint_slices(ax, df, b, c0, names, cmap, shades, mixing=True)
        mae_stamp(ax, mae)
        style(ax, ymax)
        pct_axis(ax, ymax)
        ax.set_title(title, color=INK, fontsize=7, fontweight="bold", pad=3)
        ax.set_xlabel("Teacher weight $H$", color=INK_2, fontsize=6)
        top_axes.append((bench, ax))
        if j == 0:
            ax.set_ylabel("Hint marginals\nAccuracy (%)", color=INK, fontsize=6.5,
                          fontweight="bold")

        lax = fig.add_subplot(gs[1, j])
        lax.axis("off")
        lax.legend([h for h, _ in handles], [l for _, l in handles], loc="upper center",
                   ncol=2, frameon=False, fontsize=5.3, handlelength=1.3,
                   columnspacing=0.8, labelspacing=0.25,
                   title=f"{fam_name} models", title_fontsize=5.6)

        axc = fig.add_subplot(gs[2, j])
        levels = [h for h in MIX_LEVELS if np.isclose(df["hint"], h).any()]
        capability_slices(axc, df, b, cgrid, levels, level_colour)
        mae_stamp(axc, mae)
        axc.set_xlim(cgrid[0], cgrid[-1])
        model_ticks(axc, c0, names, label_ends=True)
        style(axc, ymax)
        pct_axis(axc, ymax)
        axc.set_xlabel(CAP_LABEL.get(no_hint["capability_kind"].iloc[0], "Capability"),
                       color=INK_2, fontsize=6)
        if j == 0:
            axc.set_ylabel("Capability marginals\nAccuracy (%)", color=INK, fontsize=6.5,
                           fontweight="bold")

    benchmark_headings(fig, top_axes, fig_h)
    level_ax = fig.add_subplot(gs[3, :])
    level_ax.axis("off")
    level_ax.legend([plt.Line2D([], [], color=level_colour(h), lw=1.6) for h in MIX_LEVELS],
                    [f"{h:g}" for h in MIX_LEVELS], loc="center", ncol=len(MIX_LEVELS),
                    frameon=False, fontsize=6, handlelength=1.6,
                    title="Teacher weight $H$ (capability marginals; levels without "
                          "scores are skipped)", title_fontsize=6.5)
    fig.suptitle("Logit mixing: hint and capability marginals of the joint law", color=INK,
                 fontsize=11, fontweight="bold", y=1 - 0.12 / fig_h)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_mixing_forecasts(pairs: list, out: Path) -> list[dict]:
    """Delphi: one panel per train size t = 3 .. M - 1. Post-trained: one panel per
    (family, rule), fitted on every student, the largest model the target."""
    delphi = [p for p in pairs if p[1] == "delphi"]
    post = [p for p in pairs if p[1] != "delphi"]
    rows_of_panels = []
    for bench, fam, fam_name, _c, rule in delphi:
        no_hint, df = mix_pair_data(bench, fam, rule)
        rows_of_panels.append([(bench, fam, fam_name, rule, no_hint, df, t)
                               for t in range(3, len(no_hint))])
    for rule in ("logit_mix", "logit_mix_std"):
        row = []
        for bench, fam, fam_name, _c, r in post:
            if r == rule:
                no_hint, df = mix_pair_data(bench, fam, rule)
                row.append((bench, fam, fam_name, rule, no_hint, df, len(no_hint) - 1))
        if row:
            rows_of_panels.append(row)
    ncol = max(len(r) for r in rows_of_panels)
    fig = plt.figure(figsize=(1.45 * ncol + 1.0, 1.45 * len(rows_of_panels) + 1.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(rows_of_panels), ncol, left=0.1, right=0.985,
                          top=1 - 0.45 / fig.get_figheight(),
                          bottom=0.85 / fig.get_figheight(), wspace=0.22, hspace=0.75)
    out_rows = []
    for r, panels in enumerate(rows_of_panels):
        for j, (bench, fam, fam_name, rule, no_hint, df, t) in enumerate(panels):
            fc = forecast(no_hint, {rule: df}, t)
            out_rows.append(dict(pair=f"{bench}/{fam}", rule=rule, t=t,
                                 baseline=fc["errs"]["baseline"], law=fc["errs"]["mean"]))
            ax = fig.add_subplot(gs[r, j])
            ymax = FORECAST_YMAX.get((bench, fam), 1.0)
            draw_forecast(ax, fc, t, no_hint["model"].tolist(), ymax, first=True, faint=False)
            head = (f"{t} fitted models" if fam == "delphi"
                    else f"{BENCH_NAME[bench]} {fam_name}")
            ax.set_title(head, color=INK, fontsize=6, pad=3)
            if j == 0:
                ax.set_ylabel(f"{BENCH_NAME[bench] if fam == 'delphi' else 'AIME'}\n"
                              f"{'Delphi' if fam == 'delphi' else LABELS[rule]}",
                              color=INK, fontsize=6.5, fontweight="bold")
        for j in range(len(panels), ncol):
            fig.add_subplot(gs[r, j]).axis("off")
    hnd = [plt.Line2D([], [], color=C_BASE, lw=1.4), plt.Line2D([], [], color=C_MEAN, lw=1.4),
           plt.Line2D([], [], marker="o", ls="", ms=3.4, color=C_FIT),
           plt.Line2D([], [], marker="o", ls="", ms=3.6, color=C_HELD),
           plt.Line2D([], [], marker=CHECK, ms=6, ls="", markeredgecolor=C_WIN,
                      markerfacecolor="none", markeredgewidth=1.2, alpha=0.6)]
    fig.legend(hnd, ["Baseline", "Logit mixing law at H = 0", "Fitted models",
                     "Held-out models", "Beats baseline"],
               loc="lower center", ncol=5, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0, columnspacing=1.0)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"\n{'pair':14s} {'rule':14s} {'t':>3s} {'baseline':>9s} {'law':>9s}")
    for row in out_rows:
        win = "  beats baseline" if row["law"] < row["baseline"] else ""
        print(f"{row['pair']:14s} {row['rule']:14s} {row['t']:3d} {row['baseline']:9.4f} "
              f"{row['law']:9.4f}{win}")
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["marginals", "forecasts", "mixing", "all"])
    ap.add_argument("--pairs", default=None,
                    help="comma-separated benchmark/family pairs, e.g. "
                         "hle/qwen3,aime/gemma3; default all. `mixing` keeps only the "
                         "requested pairs that have logit mixing (gsm8k/delphi and the AIME families)")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--rollouts-dir", default=None,
                    help="directory of your own <benchmark>__<method>.csv tables (from "
                         "to_rollouts.py) in place of data/results/rollouts")
    args = ap.parse_args()
    if args.rollouts_dir:
        global ROLLOUTS
        ROLLOUTS = Path(args.rollouts_dir)

    pairs = PAIRS
    if args.pairs:
        want = [s.strip() for s in args.pairs.split(",") if s.strip()]
        known = {f"{p[0]}/{p[1]}": p for p in PAIRS}
        bad = [w for w in want if w not in known]
        if bad:
            raise SystemExit(f"unknown pair(s) {bad}; choose from {sorted(known)}")
        pairs = [known[w] for w in want]
    mix_pairs = [p for p in MIX_PAIRS if not args.pairs or f"{p[0]}/{p[1]}" in want]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.what in ("marginals", "all"):
        plot_marginals(pairs, out_dir / "marginals.pdf")
    if args.what in ("forecasts", "all"):
        report(plot_forecasts(pairs, out_dir / "forecasts.pdf"))
    if args.what in ("mixing", "all"):
        if mix_pairs:
            plot_mixing_marginals(mix_pairs, out_dir / "mixing_marginals.pdf")
            plot_mixing_forecasts(mix_pairs, out_dir / "mixing_forecasts.pdf")
        elif args.what == "mixing":
            raise SystemExit("none of the requested pairs has logit mixing")
        else:
            print("skipping mixing: none of the requested pairs has logit mixing")


if __name__ == "__main__":
    main()
