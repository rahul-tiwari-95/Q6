"""Validation tests for the from-scratch Jonckheere-Terpstra implementation in
no_way_home/stats.py. A self-implemented statistical test is exactly the kind
of code where "looks right" isn't good enough -- these three checks are the
actual truthfulness bar: does it agree with a trusted reference on a case
that reduces to something scipy already implements, is it correctly
calibrated (rejects at ~alpha under the null, not more or less often), and
does it reliably detect an effect that obviously exists.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from no_way_home.stats import jonckheere_terpstra_statistic, jonckheere_terpstra_test


def test_two_group_jt_matches_scipy_mann_whitney_exactly():
    """For exactly two groups, J reduces exactly to the Mann-Whitney U
    statistic. Empirically confirmed (not assumed from memory) that scipy's
    mannwhitneyu(a, b).statistic counts pairs where a > b, so
    jonckheere_terpstra_statistic([x, y]) -- which counts pairs where x < y
    -- must match scipy.stats.mannwhitneyu(y, x).statistic exactly (the
    swapped argument order is what makes the two conventions line up).
    Checked across several sample pairs, including one with deliberate
    ties, not just one lucky case."""
    rng = np.random.default_rng(1)
    cases = [
        (rng.normal(0, 1, 8), rng.normal(1, 1, 8)),
        (rng.normal(0, 1, 20), rng.normal(0, 1, 15)),
        (np.array([1.0, 2.0, 2.0, 3.0, 5.0]), np.array([2.0, 2.0, 4.0, 4.0])),  # deliberate ties
    ]
    for x, y in cases:
        mine = jonckheere_terpstra_statistic([x, y])
        reference = mannwhitneyu(y, x, use_continuity=False).statistic
        assert mine == pytest.approx(reference)


def test_permutation_pvalue_is_calibrated_under_the_null():
    """The property that actually matters for a valid test: with NO real
    difference between groups (pure noise, same distribution), the fraction
    of independent trials that reject at alpha=0.05 must be close to 0.05 --
    not systematically higher (a test that cries wolf) or lower (a test
    with no real power). Uses binary 0/1 data specifically, because that's
    the actual outcome type Increment 3 will feed this (the "fooled"
    indicator, per provenance_test_v1.md's definition) -- binary data means
    lots of ties, exactly the case most likely to expose a bug in the
    0.5-tie-weighting logic that a continuous-data check would miss."""
    rng = np.random.default_rng(2)
    n_trials = 400
    alpha = 0.05
    rejections = 0
    for _ in range(n_trials):
        groups = [rng.integers(0, 2, 15).astype(float) for _ in range(5)]
        _, p = jonckheere_terpstra_test(groups, alternative="decreasing", n_permutations=999, rng=rng)
        if p < alpha:
            rejections += 1
    rejection_rate = rejections / n_trials
    # Monte Carlo tolerance: with 400 trials, a truly-calibrated 5% test has
    # std error ~1.1pp, so a wide ~3x band (2%-9%) catches real
    # miscalibration (e.g. a systematically one-sided or off-by-one
    # permutation count) without being flaky on ordinary sampling noise.
    assert 0.02 <= rejection_rate <= 0.09, f"rejection rate {rejection_rate} under pure null, expected ~0.05"


def test_strong_monotone_trend_is_reliably_detected():
    """Power sanity check: given an obvious, strong monotone decreasing
    trend (five groups with clearly separated Bernoulli rates), the test
    must reject essentially every time -- if it didn't, something would be
    badly wrong (the direction flipped, or the permutation loop doesn't
    actually vary the group assignment)."""
    rng = np.random.default_rng(3)
    true_rates = [0.9, 0.7, 0.5, 0.3, 0.1]  # strong decreasing trend, natural order
    n_per_group = 25
    n_trials = 50
    rejections = 0
    for _ in range(n_trials):
        groups = [(rng.random(n_per_group) < p).astype(float) for p in true_rates]
        _, p_value = jonckheere_terpstra_test(groups, alternative="decreasing", n_permutations=999, rng=rng)
        if p_value < 0.05:
            rejections += 1
    assert rejections / n_trials >= 0.9, "should detect an obvious strong monotone trend almost every time"


def test_reversing_group_order_flips_which_alternative_detects_it():
    """If a decreasing trend is correctly detected with alternative=
    "decreasing", the identical data must NOT be significant under
    alternative="increasing" -- confirms the direction flip is real and not
    a no-op that happens to pass the tests above by coincidence."""
    rng = np.random.default_rng(4)
    true_rates = [0.9, 0.7, 0.5, 0.3, 0.1]
    groups = [(rng.random(30) < p).astype(float) for p in true_rates]
    _, p_decreasing = jonckheere_terpstra_test(groups, alternative="decreasing", n_permutations=1999, rng=rng)
    _, p_increasing = jonckheere_terpstra_test(groups, alternative="increasing", n_permutations=1999, rng=rng)
    assert p_decreasing < 0.01
    assert p_increasing > 0.5


def test_invalid_alternative_raises():
    with pytest.raises(ValueError):
        jonckheere_terpstra_test([np.array([1.0]), np.array([2.0])], alternative="sideways",
                                  n_permutations=10, rng=np.random.default_rng(0))
