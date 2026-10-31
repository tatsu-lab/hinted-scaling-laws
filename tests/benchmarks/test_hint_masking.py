from __future__ import annotations

from hinted.benchmarks import mask_word

HINT = "First compute the discriminant then take the square root of it."
N_WORDS = len(HINT.split())  # 11


def test_full_fraction_is_unchanged() -> None:
    assert mask_word(HINT, 1.0, seed="p|0") == HINT


def test_zero_fraction_masks_every_word() -> None:
    masked = mask_word(HINT, 0.0, seed="p|0")
    assert "[MASK]" in masked
    assert not any(tok != "[MASK]" for tok in masked.split())


def test_partial_fraction_keeps_ceil_visible() -> None:
    import math

    masked = mask_word(HINT, 0.5, seed="p|0")
    visible = [tok for tok in masked.split() if tok != "[MASK]"]
    assert len(visible) == math.ceil(N_WORDS * 0.5)  # 6 of 11
    assert "[MASK]" in masked


def test_deterministic_for_same_seed_and_fraction() -> None:
    assert mask_word(HINT, 0.4, seed="p|0") == mask_word(HINT, 0.4, seed="p|0")


def test_different_seed_gives_different_mask() -> None:
    # Overwhelmingly likely to differ for distinct seeds at 11 words.
    assert mask_word(HINT, 0.5, seed="p|0") != mask_word(HINT, 0.5, seed="p|1")


def test_whitespace_is_preserved() -> None:
    spaced = "alpha   beta\tgamma"
    masked = mask_word(spaced, 0.0, seed="s")
    assert masked == "[MASK]   [MASK]\t[MASK]"
