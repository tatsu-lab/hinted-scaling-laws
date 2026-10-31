"""The two-stage bits-per-byte scaling-law baseline on the Delphi ladder (appendix)."""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import set_style  # noqa: E402
from plot import (C_BASE, C_FIT, C_HELD, C_MEAN, RESULTS, fit_capability, fit_joint, law,  # noqa: E402
                  models_table, read_rollouts, rollout_table, sigmoid)

HERE = Path(__file__).resolve().parent
RUNGS = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20", "1e21", "1e22", "1e23"]
SEED = "original"
HOLDOUTS = [1, 2, 3, 4, 5]          # rungs held out from the top: t = 9 .. 5 fitted


def rung_capability() -> pd.Series:
    """log10 recorded training FLOPs per rung (models.csv), indexed by rung name."""
    meta = models_table().set_index("model")["capability"]
    return pd.Series({r: float(meta[f"delphi_{r}"]) for r in RUNGS})


# the paper's colours (Figures/07b_two_stage_mask_trunc_clipF.pdf)
C_NOREF, C_REF, C_MIX, C_EXT = "#1f4e79", "#c0392b", "#7b3fa0", "#8aa8c4"


# ----------------------------------------------------------------------------- data
@lru_cache(maxsize=None)
def unhinted() -> pd.Series:
    """Strict unhinted accuracy per rung, original seed, as a Series indexed by rung."""
    df = read_rollouts(rollout_table("gsm8k", "no_hint"))
    df = df[df.seed == SEED]
    acc = df.groupby("model").passed.mean()
    return pd.Series({r: float(acc[f"delphi_{r}"]) for r in RUNGS})


def hinted_cells(method: str) -> pd.DataFrame:
    """(capability, hint, acc) of one method on the original seed at RUNGS, plus H = 0."""
    df = read_rollouts(rollout_table("gsm8k", method))
    df = df[(df.seed == SEED) & df.model.isin([f"delphi_{r}" for r in RUNGS])]
    g = df.groupby(["model", "hint"]).passed.mean().reset_index().rename(columns={"passed": "acc"})
    g["capability"] = g.model.str.replace("delphi_", "").map(rung_capability())
    nh = unhinted()
    h0 = pd.DataFrame(dict(capability=rung_capability().to_numpy(), hint=0.0, acc=nh.to_numpy()))
    return pd.concat([g[["capability", "hint", "acc"]], h0], ignore_index=True)


def external_reference() -> pd.DataFrame:
    ref = pd.read_csv(RESULTS / "held_et_al_reference.csv")
    return ref[ref.group == "external"]


# ----------------------------------------------------------------------------- fits
def fit_power_law(flops, bpb):
    """[A, alpha, L_inf] of bpb = L_inf + A * C^-alpha, multi-start, L_inf in [0, 1]."""
    best = None
    for x0 in ([1.0, 0.2, 0.26], [1e2, 0.15, 0.20], [1e4, 0.25, 0.30], [1e3, 0.18, 0.22]):
        r = least_squares(lambda p: (p[2] + p[0] * flops ** (-p[1])) - bpb, x0=x0,
                          bounds=([0.0, 0.0, 0.0], [np.inf, 2.0, 1.0]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def power(p, C):
    return p[2] + p[0] * np.asarray(C, dtype=float) ** (-p[1])


def fit_bpb_sigmoid(bpb, acc):
    """[b1, b0] of acc = sigmoid(b1 * bpb + b0)."""
    best = None
    for x0 in ([-5.0, 2.0], [-10.0, 5.0], [-2.0, 1.0], [-20.0, 10.0]):
        r = least_squares(lambda b: sigmoid(b[0] * bpb + b[1]) - acc, x0=x0)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


# ----------------------------------------------------------------------------- figure
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    set_style()
    C = rung_capability().to_numpy(); flops = 10 ** C
    acc = unhinted().to_numpy()
    bpb_tab = pd.read_csv(RESULTS / "bpb_delphi.csv").set_index("model")
    bpb = np.array([bpb_tab.loc[f"delphi_{r}", "bits_per_byte"] for r in RUNGS])
    ext = external_reference()
    ref = pd.read_csv(RESULTS / "held_et_al_reference.csv")
    held = ref[ref.group.str.startswith("delphi")].set_index("train_flops")
    named = [float(r) for r in RUNGS]           # Held labels the rungs by their names
    d = 100 * np.abs(bpb - held.loc[named, "bits_per_byte"].to_numpy()) / held.loc[named, "bits_per_byte"].to_numpy()
    print(f"bits-per-byte vs Held (2026): mean {d.mean():.2f}%, worst {d.max():.2f}% over the ten rungs")

    hinted = {m: hinted_cells(m) for m in ("mask", "truncate", "logit_mix")}
    n, target = len(RUNGS), len(RUNGS) - 1
    fig = plt.figure(figsize=(13.0, 1.8 * len(HOLDOUTS) + 3.0), layout="constrained")
    gs = fig.add_gridspec(len(HOLDOUTS) + 1, 7, height_ratios=[1] * len(HOLDOUTS) + [1.3],
                          width_ratios=[1, 1, 1, 0.15, 1, 1, 1])
    cgrid = np.linspace(C.min() - 0.2, C.max() + 0.3, 300)
    errs = {k: [] for k in ("no reference models", "with reference models", "baseline", "hinting ensemble", "logit mixing")}
    fracs = []
    print(f"{'t':>2s} {'AE two-stage (no ref)':>22s} {'AE two-stage (ref)':>19s} {'AE baseline':>12s} {'AE ensemble':>12s} {'AE mixing':>10s}")
    for i, h in enumerate(HOLDOUTS):
        t = n - h; tr = slice(0, t)
        p = fit_power_law(flops[tr], bpb[tr])
        s_no = fit_bpb_sigmoid(bpb[tr], acc[tr])
        s_ref = fit_bpb_sigmoid(np.r_[bpb[tr], ext.bits_per_byte.to_numpy()], np.r_[acc[tr], ext.accuracy.to_numpy()])
        bpb_grid = power(p, 10 ** cgrid)
        pred_no, pred_ref = sigmoid(s_no[0] * power(p, flops[target]) + s_no[1]), sigmoid(s_ref[0] * power(p, flops[target]) + s_ref[1])
        errs["no reference models"].append(abs(pred_no - acc[target])); errs["with reference models"].append(abs(pred_ref - acc[target]))
        a_base = fit_capability(C[tr], acc[tr]); errs["baseline"].append(abs(sigmoid(a_base[0] * C[target] + a_base[1]) - acc[target]))
        preds = [law(fit_joint(hinted[m][hinted[m].capability <= C[t - 1] + 1e-9]), C[target], 0.0) for m in ("mask", "truncate")]
        errs["hinting ensemble"].append(abs(np.mean(preds) - acc[target]))
        errs["logit mixing"].append(abs(law(fit_joint(hinted["logit_mix"][hinted["logit_mix"].capability <= C[t - 1] + 1e-9]), C[target], 0.0) - acc[target]))
        fracs.append(100 * flops[t - 1] / flops[target])
        print(f"{t:2d} {errs['no reference models'][-1]:22.4f} {errs['with reference models'][-1]:19.4f} {errs['baseline'][-1]:12.4f} "
              f"{errs['hinting ensemble'][-1]:12.4f} {errs['logit mixing'][-1]:10.4f}")

        # two blocks, as in the paper: left without reference models, right with them; each block is
        # (1) FLOPs -> bits per byte, (2) bits per byte -> accuracy, (3) the two composed
        for blk, (sig, colour, name) in enumerate(((s_no, C_NOREF, "No reference models"),
                                                    (s_ref, C_REF, "With oracle reference models"))):
            c0 = 4 * blk                      # column 3 is a spacer between the blocks
            ax = fig.add_subplot(gs[i, c0])
            ax.axvspan(cgrid[0], C[t - 1], color="#f0efec", zorder=0)
            ax.plot(cgrid, bpb_grid, color=colour, lw=1.1)
            ax.plot(C[tr], bpb[tr], "o", ms=3, color=C_FIT, markeredgewidth=0)
            ax.plot(C[t:], bpb[t:], "o", ms=3, color=C_HELD, markeredgewidth=0)
            ax.set_xticks(C[[0, -1]]); ax.set_xticklabels(["3e18", "1e23"])
            if blk == 0:
                ax.set_ylabel(f"{t} fitted rungs\nbits per byte", fontweight="bold")
            if i == 0:
                ax.set_title(f"{name}\nFLOPs to bits per byte", fontsize=7.5)
            ax = fig.add_subplot(gs[i, c0 + 1])
            bg = np.linspace(min(bpb.min(), ext.bits_per_byte.min()) - 0.02, bpb.max() + 0.05, 300)
            ax.plot(bg, sigmoid(sig[0] * bg + sig[1]), color=colour, lw=1.1)
            if blk == 1:
                ax.plot(ext.bits_per_byte, ext.accuracy, "^", ms=3.5, color=C_EXT, markeredgewidth=0)
            ax.plot(bpb[tr], acc[tr], "o", ms=3, color=C_FIT, markeredgewidth=0)
            ax.plot(bpb[t:], acc[t:], "o", ms=3, color=C_HELD, markeredgewidth=0)
            ax.invert_xaxis(); ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title("bits per byte to accuracy", fontsize=7.5)
            ax = fig.add_subplot(gs[i, c0 + 2])
            ax.axvspan(cgrid[0], C[t - 1], color="#f0efec", zorder=0)
            ax.plot(cgrid, sigmoid(sig[0] * bpb_grid + sig[1]), color=colour, lw=1.1)
            ax.plot(cgrid, sigmoid(a_base[0] * cgrid + a_base[1]), color=C_BASE, lw=1.0, ls=(0, (4.5, 1.6)))
            ax.plot(C[tr], acc[tr], "o", ms=3, color=C_FIT, markeredgewidth=0)
            ax.plot(C[t:], acc[t:], "o", ms=3, color=C_HELD, markeredgewidth=0)
            ax.set_ylim(-0.02, 0.45); ax.set_xticks(C[[0, -1]]); ax.set_xticklabels(["3e18", "1e23"])
            if i == 0:
                ax.set_title("composed: FLOPs to accuracy", fontsize=7.5)
    # bottom: absolute error at 1e23 against the size of the largest fitted rung
    ax = fig.add_subplot(gs[len(HOLDOUTS), :])
    for lab, colour in (("baseline", C_BASE), ("hinting ensemble", C_MEAN), ("logit mixing", C_MIX),
                        ("no reference models", C_NOREF), ("with reference models", C_REF)):
        ax.plot(fracs, errs[lab], marker="o", ms=3, lw=1.2, color=colour, label=lab if "reference" not in lab else f"two-stage, {lab}")
    ax.set_xscale("log"); ax.set_xticks(fracs); ax.set_xticklabels([f"{v:.2g}%" for v in fracs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("largest fitted rung, % of the target's training FLOPs"); ax.set_ylabel("absolute error at 1e23")
    ax.legend(fontsize=6, ncols=3)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C_NOREF, lw=1.2, label="two-stage, no reference models"),
                        Line2D([], [], color=C_REF, lw=1.2, label="two-stage, with reference models"),
                        Line2D([], [], color=C_BASE, lw=1.0, ls=(0, (4.5, 1.6)), label="baseline: sigmoid on the H = 0 accuracies"),
                        Line2D([], [], color=C_FIT, marker="o", ls="none", ms=4, label="fitted rungs"),
                        Line2D([], [], color=C_HELD, marker="o", ls="none", ms=4, label="held-out rungs"),
                        Line2D([], [], color=C_EXT, marker="^", ls="none", ms=4, label="reference models")],
               loc="outside lower center", ncols=3, fontsize=6.5)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out = out / "two_stage.pdf"
    fig.savefig(out); plt.close(fig); print(f"wrote {out}")


if __name__ == "__main__":
    main()
