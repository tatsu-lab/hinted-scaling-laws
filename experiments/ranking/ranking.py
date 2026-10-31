"""Model ranking from hinted rollouts (paper Section 5 / Appendix G).

The truth is each model's unhinted accuracy on the holdout (`pool == "y"`). Each estimator
spends a budget of b rollouts per problem, drawn by Monte Carlo from the sampling pool
(`pool == "x"` and the hinted tables), and is scored by pairwise accuracy and Kendall tau over
the certified model pairs (Bonferroni over all pairs of the model universe):

  avg_at_k              all b rollouts at H = 0
  hinted_h<H>           all b rollouts at hint level H
  hint_prior            one hinted rollout per problem at a uniformly dealt level, b - 1 at H = 0;
                        score alpha * z(unhinted) + (1 - alpha) * z(hinted), alpha = S / (S + lambda P)
  hint_prior_lambda<L>  hint_prior over a sweep of lambda

Budgets below one rollout per problem draw a fresh problem subset per trial. The log-loss
baseline ranks models by mean bits per byte (`bpb.csv`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "results"
OUT = HERE / "out"

FRACTIONAL_BUDGETS = tuple(2.0 ** -e for e in range(7, 0, -1))
Pools = dict[str, dict[float, dict[str, np.ndarray]]]      # model -> hint -> problem -> labels


# ----------------------------------------------------------------------------- data
def load_config(path: Path | None = None) -> dict:
    """Read ranking_config.json (model universe, scored subset, hint levels, Monte-Carlo settings)."""
    return json.loads((path or DATA / "ranking_config.json").read_text())


def read_rollouts(fp: Path) -> pd.DataFrame:
    """Expand a release rollout table to one row per rollout.

    The tables hold one row per (model[, hint], problem) with `passed` a string of 0/1
    characters in rank order and, at H = 0, `pool` a parallel string of x/y/u characters. The
    position in the string is the rollout's rank, returned here as `sample_id`."""
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


def load_pools(bench: str, method: str | None, cfg: dict, data_dir: Path) -> tuple[Pools, Pools, list[str]]:
    """(sampling pools X, ground-truth pools Y, problems). At h=0 the `pool` column of
    `<bench>__no_hint` says which rollouts the estimators may sample ('x', the paper's fixed
    sampling pool) and which define the truth ('y', every other h=0 rollout). The hinted tables
    (ranks < hinted_per_problem) are sampling pools too; `method=None` loads h = 0 only."""
    b = cfg["benchmarks"][bench]
    universe = list(b["models"])
    h0 = read_rollouts(data_dir / "rollouts" / f"{bench}__no_hint.csv")
    h0 = h0[h0.model.isin(universe)]
    if method is None:
        hinted = h0.iloc[0:0].assign(hint=np.array([], dtype=float))
    else:
        hinted = read_rollouts(data_dir / "rollouts" / f"{bench}__{method}.csv")
        hinted = hinted[hinted.model.isin(universe) & (hinted.sample_id < b["hinted_per_problem"])]
    problems = sorted(h0.problem_id.astype(str).unique())
    x: Pools = {m: {} for m in universe}
    y: Pools = {m: {} for m in universe}
    for model, g in h0.groupby("model"):
        g = g.sort_values(["problem_id", "sample_id"])
        xs, ys = {}, {}
        for pid, gg in g.groupby("problem_id"):
            xs[str(pid)] = gg.passed[gg.pool == "x"].to_numpy().astype(bool)
            ys[str(pid)] = gg.passed[gg.pool == "y"].to_numpy().astype(bool)
        x[model][0.0] = xs
        y[model][0.0] = {p: v for p, v in ys.items() if v.size}
    for (model, h), g in hinted.groupby(["model", "hint"]):
        g = g.sort_values(["problem_id", "sample_id"])
        x[model][round(float(h), 1)] = {str(pid): gg.passed.to_numpy().astype(bool) for pid, gg in g.groupby("problem_id")}
    missing = [m for m in universe if 0.0 not in x[m] or not x[m][0.0]]
    if missing:
        raise SystemExit(f"{bench}: no h=0 rollouts for {missing}")
    return x, y, problems


def load_bpb(bench: str, models: list[str], problems: list[str], data_dir: Path, arm: str = "template") -> dict[str, dict[str, np.ndarray]] | None:
    """Per-model, per-problem bits-per-byte of the reference solutions from bpb.csv, or None if incomplete."""
    fp = data_dir / "bpb.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df = df[(df.benchmark == bench) & (df.arm == arm) & df.model.isin(models)]
    table: dict[str, dict[str, np.ndarray]] = {}
    for model, g in df.groupby("model"):
        per = {str(pid): gg.sort_values("ref_id").bpb.to_numpy(dtype=np.float64) for pid, gg in g.groupby("problem_id")}
        if set(per) != set(problems):
            return None
        table[model] = per
    return table if set(table) == set(models) else None


# ----------------------------------------------------------------------------- truth + certification
def truth_and_stats(y: Pools) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Ground truth per model (mean per-problem h = 0 accuracy on the holdout) with its standard error."""
    truth, stats = {}, {}
    for model, fr in y.items():
        h0 = fr.get(0.0) or {}
        rates = [float(v.mean()) for v in h0.values() if v.size]
        ks = [int(v.size) for v in h0.values() if v.size]
        if not rates:
            continue
        n = len(rates)
        truth[model] = float(np.mean(rates))
        stats[model] = {"score": truth[model], "se": math.sqrt(sum(p * (1 - p) / k for p, k in zip(rates, ks)) / (n * n)),
                        "min_k": min(ks), "max_k": max(ks), "problems": n}
    return truth, stats


def certify_pairs(stats: dict[str, dict[str, float]], alpha: float) -> tuple[set[frozenset[str]], float, int]:
    """Pairs over the whole model universe whose truth gap clears a Bonferroni-corrected z test; returns (pairs, z, J)."""
    models = sorted(stats)
    pairs = list(itertools.combinations(models, 2))
    if not pairs:
        return set(), float("nan"), 0
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * len(pairs)))
    certified = set()
    for a, b in pairs:
        delta = stats[a]["score"] - stats[b]["score"]
        se = math.hypot(stats[a]["se"], stats[b]["se"])
        if se > 0 and abs(delta) > z * se:
            certified.add(frozenset((a, b)))
    return certified, z, len(pairs)


def longest_certified_chain(ranked: list[str], certified: set[frozenset[str]]) -> list[str]:
    """Longest subsequence of the truth ranking in which every pair is certified (O(M^2) DP)."""
    best: list[list[str]] = []
    for i, m in enumerate(ranked):
        cand = [m]
        for j in range(i):
            if all(frozenset((m, o)) in certified for o in best[j]) and len(best[j]) + 1 > len(cand):
                cand = best[j] + [m]
        best.append(cand)
    return max(best, key=len) if best else []


# ----------------------------------------------------------------------------- metrics
def pairwise_accuracy(est: dict[str, float], truth: dict[str, float], pairs: set[frozenset[str]]) -> float:
    """Fraction of scored pairs an estimate orders like the truth, ties in the estimate counting one half."""
    total, score = 0, 0.0
    models = sorted(set(est) & set(truth))
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if frozenset((a, b)) not in pairs:
                continue
            td = truth[a] - truth[b]
            if td == 0:
                continue
            ed = est[a] - est[b]
            total += 1
            score += 0.5 if ed == 0 else float((ed > 0) == (td > 0))
    return score / total if total else float("nan")


def kendall_tau(est: dict[str, float], truth: dict[str, float], pairs: set[frozenset[str]]) -> float:
    """Kendall tau-a of an estimate against the truth over the scored pairs."""
    c = d = t = 0
    models = sorted(set(est) & set(truth))
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if frozenset((a, b)) not in pairs:
                continue
            td = truth[a] - truth[b]
            if td == 0:
                continue
            ed = est[a] - est[b]
            if ed == 0:
                t += 1
            elif (ed > 0) == (td > 0):
                c += 1
            else:
                d += 1
    n = c + d + t
    return (c - d) / n if n else float("nan")


def rank_metrics(est, truth, pairs) -> tuple[float, float]:
    """(pairwise accuracy, Kendall tau) of one estimate, asserting the identity tau = 2 * acc - 1."""
    acc, tau = pairwise_accuracy(est, truth, pairs), kendall_tau(est, truth, pairs)
    if not (math.isnan(acc) or math.isnan(tau)):
        assert abs(tau - (2 * acc - 1)) < 1e-9, (acc, tau)
    return acc, tau


# ----------------------------------------------------------------------------- estimators
class Method:
    """One estimator: how it spends a budget over hint levels and how it scores a sampled trial."""

    def __init__(self, name: str, label: str, kind: str, allocate: Callable[[float], dict[float, float]],
                 spread: tuple[float, ...] = (), lam: float = 8.0, seed_key: str | None = None):
        """Store the estimator's name, label, kind, allocation rule, hint levels, lambda and RNG key."""
        self.name, self.label, self.kind, self.allocate, self.spread, self.lam = name, label, kind, allocate, spread, lam
        # Each estimator draws from its own RNG stream keyed by a fixed string. The keys are the
        # estimator names of the pipeline that produced the paper's figures, kept so the Monte-Carlo
        # rows reproduce those figures to the digit; they are not meant to be read.
        self.seed_key = seed_key or name


def build_methods(hint_levels: list[float], lam: float, sweep: list[float]) -> list[Method]:
    """The estimators of one panel: avg@k, one hinted-accuracy proxy per level, hint-prior and its lambda sweep."""
    def single(f: float):
        """Allocation that spends the whole budget at one hint level."""
        return lambda b: {f: b}
    methods = [Method("avg_at_k", "avg@k", "raw", single(0.0), seed_key="baseline_h0")]
    for h in hint_levels:
        methods.append(Method(f"hinted_h{h:g}", f"hinted accuracy at H={h:.1f}", "raw", single(float(h)), seed_key=f"raw_h{h:g}"))
    spread = tuple(sorted(float(h) for h in hint_levels))

    def spread_allocate(b: float) -> dict[float, float]:
        """Hint-prior allocation: one rollout per hint level, the rest of the budget at h = 0."""
        if b < 1:
            return {f: b / len(spread) for f in spread}
        alloc = {f: 1 for f in spread}
        if int(b) > 1:
            alloc[0.0] = int(b) - 1
        return alloc
    for l in [lam] + [t for t in sweep if float(t) != float(lam)]:
        tag = int(l) if float(l).is_integer() else l
        suffix = "" if float(l) == float(lam) else f"_lambda{tag}"
        legacy = "pilotspread_pooled_zmerge" + ("" if float(l) == float(lam) else f"_tau{tag}")
        methods.append(Method("hint_prior" + suffix, f"hint-prior ranking (lambda={l:g})", "hint_prior",
                              spread_allocate, spread, float(l), seed_key=legacy))
    return methods


def can_run(method: Method, budget: float, x: Pools, models: list[str]) -> bool:
    """Whether every scored model has enough sampling-pool rollouts at every level the method needs."""
    for model in models:
        for f, k in method.allocate(max(1, int(budget))).items():
            if k == 0:
                continue
            pools = x.get(model, {}).get(f)
            if not pools:
                return False
            sizes = [int(v.size) for v in pools.values()]
            if min(sizes) <= 0 or max(sizes) < k:
                return False
    return True


def sample_counts(rng, x: Pools, models, problems, allocation: dict[float, int], trials: int):
    """Per (model, hint): (correct per trial, total) -- Binomial(min(k, pool), c/n) per problem."""
    out = {}
    for model in models:
        per = {}
        for f, k in allocation.items():
            if k <= 0:
                continue
            pools = x[model][f]
            correct = np.array([int(pools[p].sum()) for p in problems]); sizes = np.array([int(pools[p].size) for p in problems])
            k_eff = np.minimum(int(k), sizes)
            drawn = rng.binomial(k_eff[:, None], (correct / np.maximum(sizes, 1))[:, None], size=(len(problems), trials))
            per[f] = (drawn.sum(axis=0), int(k_eff.sum()))
        out[model] = per
    return out


def sample_subset(rng, x: Pools, models, problems, fractions: tuple[float, ...], budget_fraction: float, trials: int):
    """Budget < 1: round(b*P) problems per trial, one rollout each, dealt round-robin over fractions."""
    P = len(problems)
    n = max(2, int(round(budget_fraction * P)))
    rank = rng.random((trials, P)).argsort(axis=1).argsort(axis=1)
    member = rank < n
    assign = rank % len(fractions)
    out = {}
    for model in models:
        per = {}
        for i, f in enumerate(fractions):
            pools = x[model][f]
            rates = np.array([float(pools[p].mean()) for p in problems])
            mask = member & (assign == i) if len(fractions) > 1 else member
            draws = rng.random((trials, P)) < rates[None, :]
            per[f] = ((draws & mask).sum(axis=1).astype(np.int64), mask.sum(axis=1).astype(np.int64))
        out[model] = per
    return out


def sample_pilot_spread(rng, x: Pools, models, problems, hinted: tuple[float, ...], budget: float, trials: int):
    """One hinted rollout per problem (level dealt uniformly, shared permutation across models),
    budget-1 at h=0; budget < 1: a problem subset with hinted rollouts only."""
    P, F = len(problems), len(hinted)
    if budget < 1:
        n_member, h0_per = max(2, int(round(budget * P))), 0
    else:
        n_member, h0_per = P, int(budget) - 1
    rank = rng.random((trials, P)).argsort(axis=1).argsort(axis=1)
    member, assign = rank < n_member, rank % F
    out = {}
    for model in models:
        h0 = x[model][0.0]
        per = {}
        for i, f in enumerate(hinted):
            pools = x[model][f]
            rates = np.array([float(pools[p].mean()) for p in problems])
            mask = member & (assign == i)
            draws = rng.random((trials, P)) < rates[None, :]
            per[f] = ((draws & mask).sum(axis=1).astype(np.int64), mask.sum(axis=1).astype(np.int64))
        if h0_per > 0:
            sizes = np.array([int(h0[p].size) for p in problems]); rates0 = np.array([float(h0[p].mean()) for p in problems])
            k_eff = np.minimum(h0_per, sizes)
            drawn = rng.binomial(k_eff[:, None], rates0[:, None], size=(P, trials))
            per[0.0] = (drawn.sum(axis=0), np.full(trials, int(k_eff.sum()), dtype=np.int64))
        out[model] = per
    return out


def _z(values: dict[str, float]) -> dict[str, float]:
    """Standardise a dict of scores across models (zero mean, unit variance)."""
    arr = np.array(list(values.values()), dtype=np.float64)
    sd = float(arr.std())
    if sd == 0:
        return {m: 0.0 for m in values}
    mu = float(arr.mean())
    return {m: (v - mu) / sd for m, v in values.items()}


def score_trial(method: Method, sampled: dict[str, dict[float, tuple[int, int]]], P: int) -> dict[str, float]:
    """One score per model from a trial's sampled (correct, total) counts: plain accuracy, or the hint-prior z-merge."""
    if method.kind == "raw":
        return {m: (sum(c[0] for c in fc.values()) / sum(c[1] for c in fc.values())) if sum(c[1] for c in fc.values()) else 0.0
                for m, fc in sampled.items()}
    pilot, d = {}, {}
    for m, cells in sampled.items():
        hinted = [c for f, c in cells.items() if float(f) != 0.0]
        tot = sum(c[1] for c in hinted)
        pilot[m] = sum(c[0] for c in hinted) / tot if tot else 0.0
        c0 = cells.get(0.0, (0, 0))
        d[m] = c0[0] / c0[1] if c0[1] else 0.0
    s = float(np.mean([cells.get(0.0, (0, 0))[1] for cells in sampled.values()]))
    alpha = s / (s + method.lam * P)
    zd, zq = _z(d), _z(pilot)
    return {m: alpha * zd[m] + (1 - alpha) * zq[m] for m in sampled}


# ----------------------------------------------------------------------------- trials
def budget_ladder(cfg_b: dict) -> list[float]:
    """Rollouts-per-problem budgets to evaluate: fractions 2^-7..2^-1, powers of two, plus the collected depths."""
    total = int(cfg_b["x_pool"]) + int(cfg_b["hinted_per_problem"]) * len(cfg_b["hint_levels"])
    ladder, k = [], 1
    while k <= total:
        ladder.append(k); k *= 2
    for depth in (int(cfg_b["hinted_per_problem"]), int(cfg_b["x_pool"])):
        if 0 < depth <= max(ladder) and depth not in ladder:
            ladder.append(depth)
    return list(FRACTIONAL_BUDGETS) + [float(b) for b in sorted(ladder)]


def run_trials(x: Pools, models: list[str], problems: list[str], truth: dict[str, float], pairs: set[frozenset[str]],
               methods: list[Method], budgets: list[float], trials: int, seed: int) -> list[dict[str, Any]]:
    """Monte-Carlo loop: for every (method, budget), sample trials from the pools, score, and summarise tau."""
    rows = []
    P = len(problems)
    t0 = time.perf_counter()
    for method in methods:
        rng = np.random.default_rng([int(seed), zlib.crc32(method.seed_key.encode("utf-8"))])
        seen_subset: set[int] = set()
        for budget in budgets:
            if budget < 1:
                n = int(round(budget * P))
                if n < 2 or n in seen_subset:
                    continue
                seen_subset.add(n)
                budget = n / P
            if not can_run(method, max(1, int(budget)), x, models):
                continue
            if method.kind == "hint_prior":
                sampled_all = sample_pilot_spread(rng, x, models, problems, method.spread, float(budget), trials)
            elif budget < 1:
                sampled_all = sample_subset(rng, x, models, problems, tuple(method.allocate(1)), float(budget), trials)
            else:
                sampled_all = sample_counts(rng, x, models, problems, method.allocate(int(budget)), trials)
            accs, taus = [], []
            for t in range(trials):
                sampled = {m: {f: (int(c[t]), int(n[t]) if isinstance(n, np.ndarray) else int(n)) for f, (c, n) in fc.items()}
                           for m, fc in sampled_all.items()}
                est = score_trial(method, sampled, P)
                a, k = rank_metrics(est, truth, pairs)
                accs.append(a); taus.append(k)
            arr, tau_arr = np.array(accs), np.array(taus)
            sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            tsem = float(np.std(tau_arr, ddof=1) / math.sqrt(len(tau_arr))) if len(tau_arr) > 1 else 0.0
            rows.append({"method": method.name, "label": method.label, "budget": float(budget), "trials": int(len(arr)),
                         "mean_pairwise_accuracy": float(arr.mean()), "std_pairwise_accuracy": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                         "mean_ci95_lo": float(arr.mean() - 1.96 * sem), "mean_ci95_hi": float(arr.mean() + 1.96 * sem),
                         "mean_kendall_tau": float(tau_arr.mean()), "std_kendall_tau": float(np.std(tau_arr, ddof=1)) if len(tau_arr) > 1 else 0.0,
                         "tau_ci95_lo": float(tau_arr.mean() - 1.96 * tsem), "tau_ci95_hi": float(tau_arr.mean() + 1.96 * tsem)})
            print(f"  {method.name:34s} b={budget:8.4f} acc={arr.mean():.3f} tau={tau_arr.mean():.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return rows


def logloss_lines(table, models, problems, truth, pairs, seed: int) -> list[tuple[str, float, float]]:
    """(label, pairwise accuracy, tau): all-refs mean BPB ranked once; and the single-ref
    rank-then-average variant (2000 paired redraws)."""
    scores = {m: -float(np.mean([table[m][p].mean() for p in problems])) for m in models}
    acc, tau = rank_metrics(scores, truth, pairs)
    out = [("log loss (avg refs, then rank)", float(acc), float(tau))]
    rng = np.random.default_rng(int(seed))
    trials = 2000
    sums = {m: np.zeros(trials) for m in models}
    for p in problems:
        pool = int(table[models[0]][p].size)
        idx = np.argsort(rng.random((trials, pool)), axis=1)[:, :1]
        for m in models:
            sums[m] += table[m][p][idx].mean(axis=1)
    metrics = [rank_metrics({m: -(sums[m][t] / len(problems)) for m in models}, truth, pairs) for t in range(trials)]
    out.append(("log loss (rank per ref, avg acc.)", float(np.mean([a for a, _ in metrics])), float(np.mean([t for _, t in metrics]))))
    return out


# ----------------------------------------------------------------------------- main
def compute(bench: str, method: str, cfg: dict, data_dir: Path, trials: int | None = None) -> dict[str, Any]:
    """Run one (benchmark, hinting method) panel end to end and return everything the plots need."""
    b = cfg["benchmarks"][bench]
    x, y, problems = load_pools(bench, method, cfg, data_dir)
    truth, stats = truth_and_stats(y)
    ranked = sorted(truth, key=lambda m: (truth[m], m), reverse=True)
    certified, z, J = certify_pairs(stats, float(cfg["alpha"]))
    chain = longest_certified_chain(ranked, certified)
    print(f"[{bench}/{method}] universe {len(ranked)} models, {len(certified)}/{J} pairs certified (z={z:.3f}); "
          f"longest certified chain {len(chain)}/{len(ranked)}", flush=True)
    subset = list(b["model_subset"])
    pairs = {frozenset(p) for p in itertools.combinations(sorted(subset), 2)}
    uncertified = sorted(tuple(sorted(p)) for p in pairs if p not in certified)
    if uncertified:
        print(f"[{bench}/{method}] WARNING: {len(uncertified)} within-subset pair(s) NOT certified by the shipped holdout; scoring them anyway: {uncertified}", flush=True)
    models = [m for m in ranked if m in subset]
    print(f"[{bench}/{method}] scored models (truth order): {models}", flush=True)
    methods = build_methods(b["hint_levels"], float(cfg["hint_prior_lambda"]), list(cfg.get("lambda_sweep", [])))
    rows = run_trials(x, models, problems, truth, pairs, methods, budget_ladder(b), int(trials or cfg["trials"]), int(cfg["seed"]))
    table = load_bpb(bench, models, problems, data_dir)
    hlines = logloss_lines(table, models, problems, truth, pairs, int(cfg["seed"])) if table else []
    for lab, a, t in hlines:
        print(f"[{bench}/{method}] {lab}: acc={a:.3f} tau={t:.3f}")
    return {"benchmark": bench, "method": method, "models": models, "truth": {m: truth[m] for m in ranked},
            "truth_holdout_depth": {m: [stats[m]["min_k"], stats[m]["max_k"]] for m in ranked},
            "certified_pairs": len(certified), "total_pairs": J, "longest_certified_chain": chain,
            "uncertified_pairs": uncertified, "rows": rows, "logloss_hlines": hlines}


def certify(bench: str, cfg: dict, data_dir: Path, subset: list[str] | None) -> bool:
    """Print the certification report for a benchmark's model universe and judge a subset.

    Returns True when every within-subset pair is certified (the subset is the configured one
    unless `subset` is given)."""
    b = cfg["benchmarks"][bench]
    alpha = float(cfg["alpha"])
    _, y, problems = load_pools(bench, None, cfg, data_dir)
    truth, stats = truth_and_stats(y)
    ranked = sorted(truth, key=lambda m: (truth[m], m), reverse=True)
    certified, z, J = certify_pairs(stats, alpha)
    chain = longest_certified_chain(ranked, certified)
    subset = list(subset if subset is not None else b["model_subset"])
    unknown = [m for m in subset if m not in truth]
    if unknown:
        raise SystemExit(f"{bench}: subset model(s) not in the model universe or without holdout: {unknown}")

    print(f"[{bench}] universe {len(ranked)} models, {len(problems)} problems, alpha={alpha:g} over "
          f"J={J} pairs (z={z:.3f}); {len(certified)}/{J} pairs certified from the h = 0 holdout")
    print(f"\n{'truth rank':>10}  {'model':30s} {'acc':>7} {'se':>7} {'holdout k':>9} {'certified vs':>12}")
    for i, m in enumerate(ranked, 1):
        n_cert = sum(frozenset((m, o)) in certified for o in ranked if o != m)
        s = stats[m]
        k = f"{s['min_k']}" if s["min_k"] == s["max_k"] else f"{s['min_k']}-{s['max_k']}"
        tag = "  *" if m in subset else ""
        print(f"{i:>10}  {m:30s} {truth[m]:7.3f} {s['se']:7.4f} {k:>9} {n_cert:>9}/{len(ranked) - 1}{tag}")
    print(f"\nlongest fully certified chain, {len(chain)} models (every pair certified):")
    print("  " + " > ".join(chain))

    pairs = [tuple(sorted(p)) for p in itertools.combinations(sorted(subset), 2)]
    bad = [p for p in pairs if frozenset(p) not in certified]
    label = "proposed --subset" if subset != list(b["model_subset"]) else "configured model_subset"
    order = [m for m in ranked if m in subset]
    print(f"\n{label}: {len(subset)} models, {len(pairs) - len(bad)}/{len(pairs)} within-subset pairs certified")
    print("  " + " > ".join(order))
    for a_, b_ in bad:
        gap = truth[a_] - truth[b_]
        se = math.hypot(stats[a_]["se"], stats[b_]["se"])
        print(f"  NOT certified: {a_} vs {b_}  gap={gap:+.3f}  needs |gap| > {z * se:.3f}")
    print("  " + ("VALID: the subset can be scored." if not bad else
                  "INVALID: drop one model of each uncertified pair, or pick models from the chain."))
    return not bad


def compare(rows: list[dict], ref_path: Path) -> None:
    """Print the per-row tau difference between these rows and an earlier run's rows file."""
    ref = json.loads(ref_path.read_text())
    ref_rows = ref["rows"] if isinstance(ref, dict) else ref

    def modern(name: str) -> str:
        """Map the original pipeline's estimator names, if the reference rows use them, to ours."""
        name = {"baseline_h0": "avg_at_k", "pilotspread_pooled_zmerge": "hint_prior"}.get(name, name)
        name = re.sub(r"^raw_h", "hinted_h", name)
        return re.sub(r"^pilotspread_pooled_zmerge_tau", "hint_prior_lambda", name)
    key = lambda r: (modern(r["method"]), round(float(r["budget"]), 6))
    refd = {key(r): r for r in ref_rows}
    print(f"\n{'method':34s} {'budget':>8s} {'tau_ref':>8s} {'tau_new':>8s} {'diff':>7s}")
    worst = 0.0
    for r in sorted(rows, key=lambda r: (r["method"], r["budget"])):
        o = refd.get(key(r))
        if o is None:
            continue
        d = r["mean_kendall_tau"] - o["mean_kendall_tau"]; worst = max(worst, abs(d))
        print(f"{r['method']:34s} {r['budget']:8.4f} {o['mean_kendall_tau']:8.3f} {r['mean_kendall_tau']:8.3f} {d:+7.3f}")
    print(f"max |tau diff| = {worst:.4f}")


def main() -> None:
    """Command line: compute one panel, write its rows JSON to out/, optionally compare with a reference."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["rows", "certify"])
    ap.add_argument("--bench", required=True, choices=["aime", "hle"])
    ap.add_argument("--method", choices=["mask", "truncate"], help="required for `rows`")
    ap.add_argument("--subset", nargs="+", default=None, metavar="MODEL",
                    help="certify: judge this subset instead of the configured model_subset")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None, help="rows JSON from an earlier run to diff against (either naming of the estimators)")
    a = ap.parse_args()
    cfg = load_config(a.config or a.data_dir / "ranking_config.json")
    if a.what == "certify":
        sys.exit(0 if certify(a.bench, cfg, a.data_dir, a.subset) else 1)
    if a.method is None:
        ap.error("rows requires --method")
    result = compute(a.bench, a.method, cfg, a.data_dir, a.trials)
    out = a.out or (OUT / f"rows__{a.bench}__{a.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out} ({len(result['rows'])} rows)")
    if a.compare:
        compare(result["rows"], a.compare)


if __name__ == "__main__":
    main()
