"""
Tests for per-sample deterministic RNG in the CPT data pipeline.

Validates that interleaving and span generation are:
- Reproducible: same (seed, epoch, index) always produces the same output
- Order-independent: processing samples in any order gives identical per-sample results
- Epoch-sensitive: different epochs produce different outputs for the same sample index

GPU required: No.
"""

import numpy as np
import pytest

from ssi.data.cpt import get_span_idxs_binomial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng((seed, epoch, index))


# ---------------------------------------------------------------------------
# T16b: Same (seed, epoch, index) always produces same result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed, epoch, index",
    [(42, 0, 7), (42, 0, 0), (0, 0, 0), (99999, 10, 500), (42, 3, 12345)],
    ids=["default", "zero-index", "all-zeros", "large-values", "high-index"],
)
def test_span_idxs_reproducible(seed, epoch, index):
    """Calling get_span_idxs_binomial with identically-seeded RNGs produces identical output."""
    n, p, seq_len = 5, 0.4, 50
    result_a = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(seed, epoch, index))
    result_b = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(seed, epoch, index))
    assert result_a == result_b


def test_span_idxs_different_index_differs():
    """Different sample indices produce different span patterns."""
    n, p, seq_len = 5, 0.4, 50
    result_a = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(42, 0, 0))
    result_b = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(42, 0, 1))
    assert result_a != result_b


def test_span_idxs_different_epoch_differs():
    """Different epochs produce different span patterns for the same sample index."""
    n, p, seq_len = 5, 0.4, 50
    result_a = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(42, 0, 7))
    result_b = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(42, 1, 7))
    assert result_a != result_b


# ---------------------------------------------------------------------------
# T16c: Order-independent — shuffled vs sequential produces identical per-sample results
# ---------------------------------------------------------------------------


def test_span_idxs_order_independent():
    """Processing order does not affect per-sample span generation."""
    seed, epoch = 42, 3
    n, p, seq_len = 5, 0.4, 50
    indices = list(range(20))

    # Sequential
    sequential_results = {
        idx: get_span_idxs_binomial(n, p, seq_len, rng=make_rng(seed, epoch, idx))
        for idx in indices
    }

    # Shuffled (deterministic shuffle for test reproducibility)
    shuffled_indices = [13, 7, 2, 18, 0, 15, 9, 4, 11, 19, 6, 1, 16, 3, 14, 8, 17, 5, 12, 10]
    shuffled_results = {
        idx: get_span_idxs_binomial(n, p, seq_len, rng=make_rng(seed, epoch, idx))
        for idx in shuffled_indices
    }

    for idx in indices:
        assert sequential_results[idx] == shuffled_results[idx], f"Mismatch at index {idx}"


# ---------------------------------------------------------------------------
# Span structure invariants
# ---------------------------------------------------------------------------


def test_span_idxs_boundary_invariants():
    """Span indices always start at 0, end at seq_len, and are monotonically increasing."""
    n, p, seq_len = 5, 0.4, 100
    for idx in range(50):
        spans = get_span_idxs_binomial(n, p, seq_len, rng=make_rng(42, 0, idx))
        assert spans[0] == 0, f"First span index should be 0, got {spans[0]}"
        assert spans[-1] == seq_len, f"Last span index should be {seq_len}, got {spans[-1]}"
        assert spans == sorted(spans), f"Span indices should be monotonically increasing"
        assert len(spans) >= 2, "Need at least [0, seq_len]"


@pytest.mark.parametrize("seq_len", [1, 2, 5, 10, 100, 1000])
def test_span_idxs_various_seq_lengths(seq_len):
    """Span generation works for various sequence lengths without crashing."""
    spans = get_span_idxs_binomial(5, 0.4, seq_len, rng=make_rng(42, 0, 0))
    assert spans[0] == 0
    assert spans[-1] == seq_len
