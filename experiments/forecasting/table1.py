"""Table 1: in-sample MAE of the joint floored law against independent slice fits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import (fit_capability, fit_joint, law, mix_pair_data, pair_data,  # noqa: E402
                  sigmoid)

TABLE_PAIRS = [("gsm8k", "delphi"), ("aime", "qwen25"), ("aime", "gemma3"), ("hle", "qwen3"),
               ("hle", "qwen35")]
HINT_U_MIN = 0.1


def fit_sig4(x: np.ndarray, y: np.ndarray, u_min: float = HINT_U_MIN) -> np.ndarray:
    """[a1, a0, L, U] of acc = L + (U - L) sigmoid(a1 x + a0), 0 <= L <= U, u_min <= U <= 1,
    a1 >= 0. The floor is parametrised as L = t U, t in [0, 1]. Multi-start."""
    def unpack(b):
        a1, a0, U, t = b
        return a1, a0, t * U, U

    def resid(b):
        a1, a0, L, U = unpack(b)
        return L + (U - L) * sigmoid(a1 * x + a0) - y

    xm = float(np.mean(x)); ymin, ymax = float(min(y)), float(max(y))
    best = None
    for a1 in (1.0, 5.0, 20.0):
        for U, t in ((1.0, 0.0), (1.0, ymin), (max(ymax, u_min), ymin / max(ymax, 1e-6)), (0.5, 0.0)):
            U0 = min(max(U, u_min, 1e-3), 1.0); t0 = min(max(t, 0.0), 1.0)
            r = least_squares(resid, x0=[a1, -a1 * xm, U0, t0],
                              bounds=([0.0, -np.inf, u_min, 0.0], [np.inf, np.inf, 1.0, 1.0]))
            if best is None or r.cost < best.cost:
                best = r
    return np.array(unpack(best.x))


def pred_sig4(a, x):
    a1, a0, L, U = a
    return L + (U - L) * sigmoid(a1 * np.asarray(x, dtype=float) + a0)


def cell_errors(df):
    """(joint, hint-slice, capability-slice) per-model MAE vectors for one (pair, method)."""
    caps = np.sort(df["capability"].unique()); hints = np.sort(df["hint"].unique())
    b = fit_joint(df)
    joint, per_h = [], []
    for cap in caps:
        sub = df[np.isclose(df["capability"], cap)].sort_values("hint")
        h, y = sub["hint"].to_numpy(), sub["acc"].to_numpy()
        joint.append(np.abs(law(b, cap, h) - y).mean())
        per_h.append(np.abs(pred_sig4(fit_sig4(h, y), h) - y).mean())
    res_c = {cap: [] for cap in caps}
    for hl in hints:
        sub = df[np.isclose(df["hint"], hl)].sort_values("capability")
        c, y = sub["capability"].to_numpy(), sub["acc"].to_numpy()
        a = fit_capability(c, y)
        for cap, r in zip(c, np.abs(sigmoid(a[0] * c + a[1]) - y)):
            res_c[cap].append(r)
    return np.array(joint), np.array(per_h), np.array([np.mean(res_c[cap]) for cap in caps]), b


def main() -> None:
    cells = {}
    for bench, fam in TABLE_PAIRS:
        _, full = pair_data(bench, fam)
        for m in ("mask", "truncate"):
            cells[(bench, fam, m)] = full[m]
    cells[("gsm8k", "delphi", "logit_mix")] = mix_pair_data("gsm8k", "delphi", "logit_mix")[1]

    print(f"{'pair':14s} {'method':10s} {'joint':>6s} {'indiv':>6s}  ({'hint':>5s} {'cap':>5s})  "
          f"{'b1':>6s} {'b2':>6s} {'b0':>7s} {'A':>5s} {'L':>6s} {'F':>6s}")
    rows = {}
    for (bench, fam, m), df in cells.items():
        j, h, c, b = cell_errors(df)
        rows[(bench, fam, m)] = (j, h, c)
        print(f"{bench + '/' + fam:14s} {m:10s} {100 * j.mean():6.2f} {100 * (h.mean() + c.mean()) / 2:6.2f}  "
              f"({100 * h.mean():5.2f} {100 * c.mean():5.2f})  "
              f"{b[0]:6.2f} {b[1]:6.2f} {b[2]:7.2f} {b[3]:5.3f} {b[4]:6.3f} {b[5]:6.4f}")

    print("\nTable 1 (MAE in percentage points, averaged over models then over pairs)")
    print(f"{'Method':14s} {'Joint fit':>10s} {'Indiv. fit':>11s}   params joint / indiv per pair")
    for m, label in (("mask", "Masking"), ("truncate", "Truncation"), ("logit_mix", "Logit mixing")):
        J, I, npar = [], [], []
        for key, (j, h, c) in rows.items():
            if key[2] != m:
                continue
            J.append(j.mean()); I.append((h.mean() + c.mean()) / 2)
            npar.append(f"6/{4 * len(j) + 2 * cells[key]['hint'].nunique()}")
        print(f"{label:14s} {100 * np.mean(J):10.1f} {100 * np.mean(I):11.1f}   {' '.join(npar)}")


if __name__ == "__main__":
    main()
