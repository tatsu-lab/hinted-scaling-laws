"""Shared helpers for the appendix figure scripts (slices.py, contours.py, answer_mixing.py,
pass_at_k.py): the plain style, slice drawing with per-slice fits, and the forecast rows."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import (BENCH_NAME, C_BASE, C_FIT, C_HELD, C_MEAN, CAP_LABEL, HEADLINE_PAIRS,  # noqa: E402
                  PAIRS, fit_capability, fit_joint, forecast, law, sigmoid, size_tag,
                  train_sizes)
from table1 import fit_sig4, pred_sig4  # noqa: E402

HERE = Path(__file__).resolve().parent
HEADLINE = [p for p in PAIRS if f"{p[0]}/{p[1]}" in HEADLINE_PAIRS]   # the paper's five headline pairs
VIRIDIS = plt.get_cmap("viridis")


def set_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold", "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3e0", "grid.linewidth": 0.5, "axes.axisbelow": True})


def model_ticks(ax, c0, names):
    ax.set_xticks(c0); labs = [""] * len(c0); labs[0], labs[-1] = names[0], names[-1]
    ax.set_xticklabels(labs)


def pct(ax, top=1.0):
    ticks = np.arange(0, top + 1e-9, 0.2 if top > 0.45 else 0.1)
    ax.set_ylim(-0.03 * top, top * 1.03); ax.set_yticks(ticks); ax.set_yticklabels([f"{100 * t:.0f}" for t in ticks])


def draw_hint_slices(ax, df, c0, cmap, hint_fit="sig4"):
    shades = np.linspace(0.35, 0.95, len(c0))
    hgrid = np.linspace(df["hint"].min(), df["hint"].max(), 200)
    for cap, sh in zip(c0, shades):
        sub = df[np.isclose(df["capability"], cap)].sort_values("hint")
        h, y = sub["hint"].to_numpy(), sub["acc"].to_numpy()
        ax.plot(h, y, "o", ms=2.2, color=cmap(sh), markeredgewidth=0, zorder=4)
        if hint_fit == "linear":
            a = np.polyfit(h, y, 1); ax.plot(hgrid, np.polyval(a, hgrid), color=cmap(sh), lw=1.0)
        else:
            ax.plot(hgrid, pred_sig4(fit_sig4(h, y), hgrid), color=cmap(sh), lw=1.0)


def draw_cap_slices(ax, df, c0, names, cap_fit="pinned"):
    hints = np.sort(df["hint"].unique()); hmax = hints.max()
    cgrid = np.linspace(c0.min() - 0.2, c0.max() + 0.2, 200)
    for hl in hints:
        sub = df[np.isclose(df["hint"], hl)].sort_values("capability")
        c, y = sub["capability"].to_numpy(), sub["acc"].to_numpy()
        color = VIRIDIS(hl / hmax if hmax > 0 else 0)
        ax.plot(c, y, "o", ms=2.2, color=color, markeredgewidth=0, zorder=4)
        if cap_fit == "free":
            ax.plot(cgrid, pred_sig4(fit_sig4(c, y, u_min=0.0), cgrid), color=color, lw=1.0)
        else:
            a = fit_capability(c, y); ax.plot(cgrid, sigmoid(a[0] * cgrid + a[1]), color=color, lw=1.0)
    model_ticks(ax, c0, names)


def draw_joint_slices(ah, ac, df, c0, names, cmap):
    """The floored law fitted to every cell of the arm, drawn as its hint slices (one per
    model) and capability slices (one per level), with the cells as dots."""
    b = fit_joint(df)
    shades = np.linspace(0.35, 0.95, len(c0)); hints = np.sort(df["hint"].unique()); hmax = hints.max()
    hgrid = np.linspace(0, hmax, 200); cgrid = np.linspace(c0.min() - 0.2, c0.max() + 0.2, 200)
    for cap, sh in zip(c0, shades):
        sub = df[np.isclose(df["capability"], cap)].sort_values("hint")
        ah.plot(sub["hint"], sub["acc"], "o", ms=2.2, color=cmap(sh), markeredgewidth=0, zorder=4)
        ah.plot(hgrid, law(b, cap, hgrid), color=cmap(sh), lw=1.0)
    for hl in hints:
        sub = df[np.isclose(df["hint"], hl)].sort_values("capability")
        color = VIRIDIS(hl / hmax if hmax > 0 else 0)
        ac.plot(sub["capability"], sub["acc"], "o", ms=2.2, color=color, markeredgewidth=0, zorder=4)
        ac.plot(cgrid, law(b, cgrid, hl), color=color, lw=1.0)
    model_ticks(ac, c0, names)


def draw_forecast_rows(fig_title, pairs, arms, hint_label, out: Path, slice_hint_fit, slice_cap_fit,
                       slices: str = "individual", extras: dict | None = None, colour: str = C_MEAN,
                       ae_panel: bool = True, train_sizes_fn=None) -> None:
    """Top: the arm's hint and capability slices per pair, each fitted alone (`slices` =
    "individual") or as slices of the joint law fitted to every cell ("joint"). Bottom: the
    floored law fitted to the arm on the t smallest models, its H = 0 slice against the
    baseline sigmoid, one row per pair, plus the absolute error at the largest model.

    `extras` adds comparison forecasts per pair: {(bench, fam): [(label, colour, no_hint, full)]},
    each drawn as the mean H = 0 forecast of the laws fitted to `full`'s methods on the same
    t smallest models (one method: that law; two: their ensemble). `colour` is the arm's own
    forecast colour; `ae_panel` adds the absolute-error panel at the end of each row;
    `train_sizes_fn(benchmark, n_models)` overrides plot.train_sizes. A legend
    under the figure names every curve and marker."""
    n = len(pairs)
    tsets = {k: (train_sizes_fn or train_sizes)(k[0], len(v[0])) for k, v in arms.items()}
    ncol = max(len(v) for v in tsets.values())
    fig = plt.figure(figsize=(1.6 * (ncol + int(ae_panel)) + 1.2, 1.8 * 2 + 1.6 * n + 1.0), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.55 * n])
    top = outer[0].subgridspec(2, n); bot = outer[1].subgridspec(n, ncol + int(ae_panel))
    for r, (bench, fam, flab, cmap_name) in enumerate(pairs):
        no_hint, df = arms[(bench, fam)]
        c0, y0 = no_hint["capability"].to_numpy(), no_hint["acc"].to_numpy()
        names = [size_tag(m) for m in no_hint["model"]]
        ah, ac = fig.add_subplot(top[0, r]), fig.add_subplot(top[1, r])
        if slices == "joint":
            draw_joint_slices(ah, ac, df, c0, names, plt.get_cmap(cmap_name))
        else:
            draw_hint_slices(ah, df, c0, plt.get_cmap(cmap_name), hint_fit=slice_hint_fit)
            draw_cap_slices(ac, df, c0, names, cap_fit=slice_cap_fit)
        pct(ah); pct(ac); ah.set_title(f"{BENCH_NAME[bench]} {flab}"); ah.set_xlabel(hint_label)
        ac.set_xlabel(CAP_LABEL[no_hint["capability_kind"].iloc[0]])
        if r == 0:
            ah.set_ylabel("Accuracy (%)"); ac.set_ylabel("Accuracy (%)")
        ymax = {("gsm8k", "delphi"): 0.4, ("hle", "qwen3"): 0.4, ("hle", "qwen35"): 0.6}.get((bench, fam), 1.0)
        errs = []
        extra = (extras or {}).get((bench, fam), [])
        extra_errs = {lab: [] for lab, *_ in extra}
        for j, t in enumerate(tsets[(bench, fam)]):
            fc = forecast(no_hint, {"arm": df}, t)
            ax = fig.add_subplot(bot[r, j])
            ax.axvspan(fc["cgrid"][0], fc["boundary"], color="#f0efec", zorder=0)
            ax.plot(fc["cgrid"], fc["base_curve"], color=C_BASE, lw=1.2, ls=(0, (4.5, 1.6)))
            for lab, xcolour, nh_x, full_x in extra:
                # the comparison law(s) are fitted on the same t smallest models of this pair
                fx = forecast(nh_x, full_x, t)
                ax.plot(fx["cgrid"], fx["mean_curve"], color=xcolour, lw=1.1, ls=(0, (3.0, 1.2, 1.0, 1.2)))
                extra_errs[lab].append(fx["errs"]["mean"])
            ax.plot(fc["cgrid"], fc["mean_curve"], color=colour, lw=1.4, ls=(0, (4.5, 1.6)), zorder=4)
            ax.plot(c0[:t], y0[:t], "o", ms=3, color=C_FIT, markeredgewidth=0, zorder=5)
            ax.plot(c0[t:], y0[t:], "o", ms=3, color=C_HELD, markeredgewidth=0, zorder=5)
            pct(ax, ymax); model_ticks(ax, c0, names); ax.set_title(f"{t} fitted models", fontweight="normal", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{BENCH_NAME[bench]} {flab}\nAccuracy (%)", fontweight="bold")
            else:
                ax.tick_params(labelleft=False)
            errs.append((fc["errs"]["baseline"], fc["errs"]["mean"]))
        if not ae_panel:
            continue
        axm = fig.add_subplot(bot[r, ncol])
        frac = [100 * 10 ** (c0[t - 1] - c0[-1]) for t in tsets[(bench, fam)]]
        axm.plot(frac, [e[0] for e in errs], marker="o", ms=2.5, color=C_BASE, lw=1.1, label="baseline")
        axm.plot(frac, [e[1] for e in errs], marker="o", ms=2.5, color=colour, lw=1.1, label=fig_title)
        for lab, xcolour, *_ in extra:
            axm.plot(frac, extra_errs[lab], marker="o", ms=2.5, color=xcolour, lw=1.1, label=lab)
        axm.set_xscale("log"); axm.set_title("AE, largest model", fontweight="normal", fontsize=7)
        axm.set_xticks(frac); axm.set_xticklabels([f"{v:.2g}%" if v < 10 else f"{v:.0f}%" for v in frac], fontsize=6)
        axm.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axm.set_xlabel("largest fitted model, % of target scale")
    # one legend for the whole figure: every forecast curve and the two kinds of model
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_BASE, lw=1.2, ls=(0, (4.5, 1.6)), label="baseline: sigmoid on the H = 0 accuracies"),
               Line2D([], [], color=colour, lw=1.4, ls=(0, (4.5, 1.6)), label=fig_title)]
    seen = set()
    for extra in (extras or {}).values():
        for lab, xcolour, *_ in extra:
            if lab not in seen:
                seen.add(lab); handles.append(Line2D([], [], color=xcolour, lw=1.1, ls=(0, (3.0, 1.2, 1.0, 1.2)), label=lab))
    handles += [Line2D([], [], color=C_FIT, marker="o", ls="none", ms=4, label="fitted models"),
                Line2D([], [], color=C_HELD, marker="o", ls="none", ms=4, label="held-out models")]
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 3), fontsize=6.5)
    fig.savefig(out); plt.close(fig); print(f"wrote {out}")


