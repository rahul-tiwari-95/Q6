"""Jonckheere-Terpstra ordered-alternative trend test, per ENVIRONMENT_REDESIGN.md
§3 -- the pre-registered primary test for Increment 3's beta-sweep experiment.

No off-the-shelf implementation exists in this environment: scipy 1.13.1 (the
only stats package installed here) has no jonckheere/jt function, and nothing
like pingouin or scikit-posthocs is available either. This is a from-scratch
implementation, so it is validated three separate ways in tests/test_stats.py
rather than trusted on the strength of the formula alone -- see that file for
why each check matters. Do not add a normal-approximation p-value: the redesign
doc explicitly calls for a "permutation-simulated JT null," which is what
jonckheere_terpstra_test implements.
"""

from __future__ import annotations

import numpy as np


def _mann_whitney_u(x: np.ndarray, y: np.ndarray) -> float:
    """U statistic: count of pairs (xi, yj) with xi < yj, ties counting 0.5.
    Direct O(n*m) count via broadcasting -- group sizes here are tens, not
    thousands, so the O(n log n) rank-based shortcut buys nothing."""
    x = np.asarray(x)[:, None]
    y = np.asarray(y)[None, :]
    less = (x < y).sum()
    ties = (x == y).sum()
    return float(less) + 0.5 * float(ties)


def jonckheere_terpstra_statistic(groups: list) -> float:
    """J = sum over all i < j of U(group_i, group_j). Groups must already be
    in the order the trend is hypothesized to run in (group 0 hypothesized
    stochastically smallest, last group hypothesized largest) -- this
    function does not know about "increasing" vs "decreasing", that's
    jonckheere_terpstra_test's job."""
    j_stat = 0.0
    groups = [np.asarray(g) for g in groups]
    for i in range(len(groups)):
        for k in range(i + 1, len(groups)):
            j_stat += _mann_whitney_u(groups[i], groups[k])
    return j_stat


def jonckheere_terpstra_test(groups: list, alternative: str, n_permutations: int,
                              rng: np.random.Generator, *, paired: bool = False) -> tuple:
    """Permutation test for a monotone trend across len(groups) ordered
    groups, GIVEN IN THEIR NATURAL ORDER (e.g. beta=0, 0.25, ..., 1.0)
    regardless of which direction the trend is hypothesized to run --
    alternative="decreasing" flips the comparison internally rather than
    asking the caller to reverse the list themselves, which is an easy
    mistake to make silently and get a confidently-wrong p-value from.

    With paired=True, corresponding indices are seed blocks and labels are
    permuted only within each block. Otherwise samples are independent.
    Returns (J, p_value), with group order reversed for decreasing J; larger
    values support the specified alternative. Historical J used this same
    convention, despite an incorrect original docstring.
    """
    if alternative not in ("increasing", "decreasing"):
        raise ValueError(f"alternative must be 'increasing' or 'decreasing', got {alternative!r}")

    sizes = [len(g) for g in groups]
    if len(sizes) < 2 or min(sizes) == 0:
        raise ValueError("at least two nonempty groups are required")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if paired and len(set(sizes)) != 1:
        raise ValueError("paired groups must contain the same number of observations")
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in groups])

    def statistic(flat: np.ndarray) -> float:
        regrouped = []
        idx = 0
        for n in sizes:
            regrouped.append(flat[idx:idx + n])
            idx += n
        if alternative == "decreasing":
            regrouped = regrouped[::-1]
        return jonckheere_terpstra_statistic(regrouped)

    observed = statistic(pooled)
    permuted = np.empty(n_permutations)
    for p in range(n_permutations):
        if paired:
            blocks = pooled.reshape(len(sizes), sizes[0])
            order = rng.random(blocks.shape).argsort(axis=0)
            sample = np.take_along_axis(blocks, order, axis=0).ravel()
        else:
            sample = rng.permutation(pooled)
        permuted[p] = statistic(sample)
    # +1 / +1 (not the raw fraction): the standard Monte Carlo permutation
    # p-value correction (Davison & Hinkley 1997) -- avoids ever reporting
    # p=0, which is not a defensible claim from a finite number of draws.
    p_value = (np.sum(permuted >= observed) + 1) / (n_permutations + 1)
    return float(observed), float(p_value)
