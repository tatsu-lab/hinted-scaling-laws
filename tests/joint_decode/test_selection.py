from __future__ import annotations

import random

import pytest

from hinted.joint_decode.selection import select_add_logits, select_top_rank


def _topk(pairs: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    return [{"token_id": token_id, "logit": logit} for token_id, logit in pairs]


def test_select_top_rank_prefers_highest_b_rank_among_a_topk() -> None:
    a = _topk([(1, 5.0), (2, 4.0), (3, 3.0)])
    b = _topk([(9, 9.0), (3, 8.0), (2, 7.0)])

    assert select_top_rank(a, b, rng=random.Random(0), request_index=0) == 3


def test_select_top_rank_falls_back_to_a_top1_without_overlap() -> None:
    a = _topk([(1, 5.0), (2, 4.0)])
    b = _topk([(8, 9.0), (9, 8.0)])

    assert select_top_rank(a, b, rng=random.Random(0), request_index=0) == 1


def test_select_top_rank_rejects_empty_a() -> None:
    with pytest.raises(ValueError, match="empty top-k"):
        select_top_rank([], _topk([(1, 1.0)]), rng=random.Random(0), request_index=0)


def test_select_add_logits_greedy_argmax_at_zero_temperature() -> None:
    # score = a + 1.0 * b: token 1 scores 10, token 2 scores 10.5
    a = _topk([(1, 10.0), (2, 0.5)])
    b = _topk([(2, 10.0), (1, 0.0)])

    tokens_a, tokens_b = select_add_logits(
        a, b, advisor_weight=1.0, temperature=0.0, rng=random.Random(0), request_index=0
    )

    assert tokens_a == tokens_b == [2]


def test_select_add_logits_at_zero_weight_is_plain_model() -> None:
    a = _topk([(1, 10.0), (2, 0.0)])
    b = _topk([(2, 100.0), (1, 0.0)])

    tokens_a, _ = select_add_logits(
        a, b, advisor_weight=0.0, temperature=0.0, rng=random.Random(0), request_index=0
    )

    assert tokens_a == [1]


def test_select_add_logits_missing_token_takes_side_floor() -> None:
    # token 3 is only in B's top-k; on A it takes A's floor (0.0)
    a = _topk([(1, 1.0), (2, 0.0)])
    b = _topk([(3, 5.0), (1, 0.0)])

    tokens_a, _ = select_add_logits(
        a, b, advisor_weight=1.0, temperature=0.0, rng=random.Random(0), request_index=0
    )

    assert tokens_a == [3]


def test_select_add_logits_std_rescales_teacher_to_model() -> None:
    # model std 1, teacher std 10: raw lets the teacher's 20-point lead win,
    # std rescaling scales it by 0.1 so the model's 2-point lead survives
    a = _topk([(1, 2.0), (2, 0.0)])
    b = _topk([(2, 20.0), (1, 0.0)])
    kw = dict(advisor_weight=0.5, temperature=0.0, rng=random.Random(0), request_index=0)

    assert select_add_logits(a, b, **kw)[0] == [2]
    assert select_add_logits(a, b, teacher_scale="std", **kw)[0] == [1]


def test_select_add_logits_std_at_zero_weight_is_plain_model() -> None:
    a = _topk([(1, 3.0), (2, 2.5), (3, 1.0)])
    b = _topk([(3, 30.0), (4, 5.0), (1, 0.0)])
    for seed in range(50):
        raw = select_add_logits(a, b, advisor_weight=0.0, temperature=0.7,
                                rng=random.Random(seed), request_index=0)
        std = select_add_logits(a, b, advisor_weight=0.0, temperature=0.7,
                                rng=random.Random(seed), request_index=0, teacher_scale="std")
        assert raw == std


def test_select_add_logits_rejects_unknown_teacher_scale() -> None:
    with pytest.raises(ValueError, match="teacher_scale"):
        select_add_logits(_topk([(1, 1.0)]), _topk([(1, 1.0)]), advisor_weight=1.0,
                          temperature=0.0, rng=random.Random(0), request_index=0,
                          teacher_scale="gap")
