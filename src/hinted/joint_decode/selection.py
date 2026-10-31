from __future__ import annotations

import math
import random
from typing import Any


def select_add_logits(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    rng: random.Random,
    request_index: int,
    teacher_scale: str = "none",
) -> tuple[list[int], list[int]]:
    """Logit mixing: score = logit_a + alpha * logit_b.

    The decoder (side A) keeps FULL weight and the advisor (side B, the
    teacher) is added on top, so `advisor_weight` is an unbounded bias
    strength rather than a mixing weight -- the paper's grid runs 0 to 16 --
    and alpha = 0 is plain decoder sampling.

    Candidates are the union of the two sides' top-k tokens; a token missing
    from one side takes that side's top-k floor logit.

    teacher_scale="std" (std rescaling): the advisor's logits are first
    rescaled to the decoder's scale at this step,
        score = logit_a + alpha * (sigma_a / sigma_b) * logit_b,
    sigma = the plain (equal-weight, divide-by-n) standard deviation of that
    side's top-k logits. alpha then sets the same relative pull whatever the
    decoder's sharpness.
    """
    del request_index
    if advisor_weight < 0.0:
        raise ValueError("advisor_weight must be >= 0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    a_logits = {int(t["token_id"]): float(t["logit"]) for t in a_topk}
    b_logits = {int(t["token_id"]): float(t["logit"]) for t in b_topk}
    if not a_logits or not b_logits:
        raise ValueError("both sides must provide at least one top-k logit")

    if teacher_scale == "std":
        sigma_b = _std(b_logits.values())
        if sigma_b > 0.0:
            scale = _std(a_logits.values()) / sigma_b
            b_logits = {token_id: scale * logit for token_id, logit in b_logits.items()}
    elif teacher_scale != "none":
        raise ValueError(f"teacher_scale must be 'none' or 'std', got {teacher_scale!r}")
    a_floor = min(a_logits.values())
    b_floor = min(b_logits.values())
    union = list(set(a_logits) | set(b_logits))
    scores = [
        a_logits.get(token_id, a_floor) + advisor_weight * b_logits.get(token_id, b_floor)
        for token_id in union
    ]
    if temperature == 0.0:
        token = union[scores.index(max(scores))]
        return [token], [token]
    max_score = max(scores)
    weights = [math.exp((score - max_score) / temperature) for score in scores]
    token = rng.choices(union, weights=weights, k=1)[0]
    return [token], [token]


def _std(values) -> float:
    values = list(values)
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def select_top_rank(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    rng: random.Random,
    request_index: int,
) -> int:
    """Pick the token from A's top-k with the highest rank in B's top-k;
    fall back to A's top-1 when the lists do not overlap. Deterministic."""
    del rng, request_index
    a_ids = [int(t["token_id"]) for t in a_topk]
    if not a_ids:
        raise ValueError("empty top-k from side A; ensure top_k_a >= 1")
    b_rank = {int(t["token_id"]): index for index, t in enumerate(b_topk)}
    overlap = [(b_rank[token_id], token_id) for token_id in a_ids if token_id in b_rank]
    if not overlap:
        return a_ids[0]
    return min(overlap)[1]
