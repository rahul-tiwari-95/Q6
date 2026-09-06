"""Monte Carlo power calculation for Increment 3's confirmatory N, per
ENVIRONMENT_REDESIGN.md §3 ("N ... must be confirmed by an actual power
calculation ... before locking, not assumed by continuity with prior runs").

Uses run_pilot_beta_sweep.py's per-arm F(beta) as working parameters, with
Jeffreys/Laplace smoothing ((successes + 0.5) / (n + 1)) rather than the raw
0/20 and 20/20 estimates the pilot actually produced -- plugging in a literal
0.0 or 1.0 probability would make this power calculation overconfident by
construction (a true rate of, say, 5% can easily produce zero successes in
20 draws by chance; smoothing accounts for that instead of pretending the
pilot proved certainty it didn't).

For each candidate N, simulates n_replicates synthetic datasets by drawing
Bernoulli(smoothed_p_beta) outcomes for N seeds per arm, runs the JT
permutation test on each, and reports the fraction that reject at alpha=0.05
-- that fraction IS the estimated power at that N, given the smoothed pilot
effect size.

Usage: python3 -m no_way_home.run_power_calculation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from no_way_home.stats import jonckheere_terpstra_test

# From run_pilot_beta_sweep.py's actual output (results/pilot_beta_sweep_v1.md):
# beta:        0.0   0.25  0.5   0.75  1.0
# fooled/N:    13/20 12/20 10/20 0/20  0/20
PILOT_SUCCESSES = [13, 12, 10, 0, 0]
PILOT_N = 20

CANDIDATE_NS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
N_REPLICATES = 500
N_PERMUTATIONS_PER_REPLICATE = 999
ALPHA = 0.05
TARGET_POWER = 0.8


def smoothed_probabilities() -> list:
    """Jeffreys-ish smoothing: (successes + 0.5) / (n + 1). Keeps every arm's
    probability strictly inside (0, 1) so the simulation never assumes
    certainty the pilot's finite sample didn't actually establish."""
    return [(s + 0.5) / (PILOT_N + 1) for s in PILOT_SUCCESSES]


def estimate_power(probabilities: list, n_per_arm: int, rng: np.random.Generator) -> float:
    rejections = 0
    for _ in range(N_REPLICATES):
        groups = [(rng.random(n_per_arm) < p).astype(float) for p in probabilities]
        _, p_value = jonckheere_terpstra_test(groups, alternative="decreasing",
                                               n_permutations=N_PERMUTATIONS_PER_REPLICATE, rng=rng)
        if p_value < ALPHA:
            rejections += 1
    return rejections / N_REPLICATES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("no_way_home/results/power_calculation_v1.md"))
    args = parser.parse_args()

    probs = smoothed_probabilities()
    rng = np.random.default_rng(123)

    lines = []
    lines.append("# No Way Home — Power Calculation for Increment 3's Confirmatory N\n")
    lines.append("Smoothed per-arm probabilities used as the working effect size, from "
                 "run_pilot_beta_sweep.py's actual output (Jeffreys smoothing, "
                 "(successes + 0.5) / (n + 1) -- NOT the raw pilot MLEs, since two arms hit "
                 "exactly 0/20 and plugging in a literal 0.0 would make this calculation "
                 "overconfident by assuming certainty a 20-seed pilot cannot establish):\n")
    lines.append("| beta | pilot F(beta) | smoothed p |")
    lines.append("|---:|---:|---:|")
    for beta, s, p in zip([0.0, 0.25, 0.5, 0.75, 1.0], PILOT_SUCCESSES, probs):
        lines.append(f"| {beta} | {s}/{PILOT_N} | {p:.3f} |")

    lines.append(f"\n## Estimated power vs. N (seeds/arm)\n")
    lines.append(f"{N_REPLICATES} Monte Carlo replicates per N, {N_PERMUTATIONS_PER_REPLICATE} "
                 f"permutations per replicate's JT test, alpha={ALPHA}.\n")
    lines.append("| N/arm | Estimated power |")
    lines.append("|---:|---:|")

    recommended_n = None
    for n in CANDIDATE_NS:
        power = estimate_power(probs, n, rng)
        marker = ""
        if power >= TARGET_POWER and recommended_n is None:
            recommended_n = n
            marker = f" **(first to reach {TARGET_POWER:.0%})**"
        lines.append(f"| {n} | {power:.2f}{marker} |")
        print(f"N={n}: power={power:.2f}")

    lines.append(f"\n## Why 'smallest N for {TARGET_POWER:.0%} power' is the wrong number to act on\n")
    lines.append(f"The mechanical answer above (N={recommended_n}) only asks 'will the JT test detect "
                 f"THAT a trend exists.' It says nothing about the actual deliverable "
                 f"(ENVIRONMENT_REDESIGN.md §3): locating the knee, "
                 f"`min(beta such that F(beta) <= 0.10)`. That's an ESTIMATION question, not a "
                 f"detection question, and small N is badly underpowered for it even when it "
                 f"clears the detection bar. Wilson 95% upper confidence bound on the true rate, "
                 f"given 0 observed successes at each N:\n")
    lines.append("| N/arm | Wilson upper bound on true F(beta), given 0 observed |")
    lines.append("|---:|---:|")
    z = 1.959964
    for n in CANDIDATE_NS:
        upper = z**2 / (n + z**2)
        lines.append(f"| {n} | {upper:.1%} |")
    lines.append(f"\nAt N=5, observing 0/5 'fooled' is still consistent with a true rate as high as "
                 f"43% -- essentially uninformative about whether beta=0.75 is actually safe. At "
                 f"N=20, that tightens to 16%; at N=30, 11%. The detection-power table above and "
                 f"this precision table are answering two different questions, and only the second "
                 f"one is close to what Increment 3 actually needs to report.")

    lines.append(f"\n## Decision\n")
    lines.append(f"**Not N={recommended_n}.** Rahul signed off on **N=50 seeds/arm**, past the "
                 f"initial N=30 suggestion, specifically for tighter precision on the "
                 f"knee-location given how sharp the pilot's step-function shape looked (a hunch "
                 f"worth respecting: a genuinely sharp transition is exactly the case where you "
                 f"want enough resolution to trust *where* it sits, not just that it exists). "
                 f"N=50 clears detection power with overwhelming margin against the pilot's "
                 f"smoothed effect size, and gets the Wilson upper bound on a zero-observed arm "
                 f"down to the value in the table above for N=50. Locked in "
                 f"PREREGISTRATION_INCREMENT3.md -- this script's own recommendation logic above "
                 f"is left as-is (mechanical smallest-N and the precision table) since it's "
                 f"useful context, but the actual decision is N=50, not whatever this function "
                 f"would output on its own.")

    lines.append(f"\n**Honest caveats**: (1) this is a power/precision calculation against the "
                 f"PILOT's own observed effect size, not a hypothesis-free calculation -- if the "
                 f"true effect is smaller than what a 20-seed pilot happened to show (regression "
                 f"to the mean is a real risk, especially for the two 0/20 arms), even N=30 could "
                 f"be optimistic. (2) The pilot's shape (a near step-function between beta=0.5 and "
                 f"beta=0.75, not a smooth gradient) is itself only a 20-seed observation and could "
                 f"look different with more data -- it's a reason to be interested in the real "
                 f"result, not a preview of it.")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
