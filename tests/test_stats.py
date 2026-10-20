"""eval/stats.py için pytest testleri."""

import numpy as np
import pytest

from tool_calling_ft.eval.stats import (
    bootstrap_ci,
    bootstrap_ci_from_counts,
    counts_to_values,
    paired_bootstrap_diff,
    unpaired_bootstrap_diff_from_counts,
    wilson_ci,
)


def test_counts_to_values():
    values = counts_to_values(3, 5)
    assert values.sum() == 3
    assert len(values) == 5

    with pytest.raises(ValueError):
        counts_to_values(6, 5)
    with pytest.raises(ValueError):
        counts_to_values(0, 0)


def test_bootstrap_ci_contains_estimate():
    ci = bootstrap_ci_from_counts(37, 80)
    assert ci.estimate == pytest.approx(37 / 80)
    assert ci.low < ci.estimate < ci.high
    assert 0.0 <= ci.low and ci.high <= 1.0
    assert ci.n == 80


def test_bootstrap_ci_is_reproducible():
    a = bootstrap_ci_from_counts(11, 80, seed=7)
    b = bootstrap_ci_from_counts(11, 80, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_ci_narrows_with_more_samples():
    small = bootstrap_ci_from_counts(10, 20)
    large = bootstrap_ci_from_counts(500, 1000)
    assert (large.high - large.low) < (small.high - small.low)


def test_bootstrap_ci_fractional_values():
    ci = bootstrap_ci([0.0, 0.5, 1.0, 0.5])
    assert ci.estimate == pytest.approx(0.5)
    assert ci.low <= 0.5 <= ci.high


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_wilson_handles_perfect_score():
    # Bootstrap 20/20 için sıfır genişlikte aralık verir; Wilson vermemeli
    assert bootstrap_ci_from_counts(20, 20).low == 1.0
    ci = wilson_ci(20, 20)
    assert ci.high == pytest.approx(1.0)
    assert ci.low < 0.9


def test_wilson_known_value():
    # 19/20 için %95 Wilson aralığı ≈ [0.764, 0.991]
    ci = wilson_ci(19, 20)
    assert ci.low == pytest.approx(0.764, abs=0.005)
    assert ci.high == pytest.approx(0.991, abs=0.005)


def test_wilson_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        wilson_ci(5, 10, confidence=0.8)


def test_paired_diff_detects_clear_difference():
    a = np.ones(80)
    b = np.concatenate([np.ones(10), np.zeros(70)])
    diff = paired_bootstrap_diff(a, b)
    assert diff.estimate == pytest.approx(70 / 80)
    assert diff.low > 0


def test_paired_diff_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1, 0, 1])


def test_unpaired_diff_small_gap_not_significant():
    # DoRA (11/80) vs LoRA (7/80): fark gürültü içinde olmalı
    diff = unpaired_bootstrap_diff_from_counts(11, 80, 7, 80)
    assert diff.estimate == pytest.approx(4 / 80)
    assert diff.low <= 0 <= diff.high


def test_unpaired_diff_large_gap_significant():
    # Full FT (74/80) vs QLoRA (37/80)
    diff = unpaired_bootstrap_diff_from_counts(74, 80, 37, 80)
    assert diff.low > 0


def test_format_and_overlap():
    a = bootstrap_ci_from_counts(7, 80)
    b = bootstrap_ci_from_counts(11, 80)
    c = bootstrap_ci_from_counts(74, 80)
    assert a.format_pct().startswith("%8.8")
    assert a.overlaps(b)
    assert not a.overlaps(c)
