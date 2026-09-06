"""PILOT (non-confirmatory) beta-sweep for Increment 3, per ENVIRONMENT_REDESIGN.md
§3 and ENGINEERING_NWH_PHASE_PLAN.md. This does NOT produce the confirmatory
result -- it exists only to get real per-arm variance and effect-size numbers
to feed a power calculation, so the eventual confirmatory N isn't picked
blind. The pilot seeds below (9000+) must never be reused as confirmatory
seeds once N is locked -- using them twice would let this pilot's own noise
leak into the "confirmatory" result.

instrument_mixed(beta) is run directly as a policy_fn (mirroring how
C3b_lineage_naive_heuristic / C3c_lineage_aware_heuristic are used directly
in POLICIES, not gated by any mandate/election machinery) -- this matches
the structure of the original 0/30-vs-24/30 comparison in
provenance_test_v1.md exactly, just with beta as a free parameter instead of
two hand-written policies.

"Fooled" uses provenance_test_v1.md's own definition, verbatim: mitigated at
least once during the pre-shift (blight_high=False) period.

Usage: python3 -m no_way_home.run_pilot_beta_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

from no_way_home.institutions import instrument_mixed
from no_way_home.stats import jonckheere_terpstra_test
from no_way_home.world import WorldConfig, run

BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_PILOT_SEEDS = 20
PILOT_SEEDS = list(range(9000, 9000 + N_PILOT_SEEDS))


def was_fooled(state) -> bool:
    """provenance_test_v1.md's exact definition: mitigated at least once
    while blight was still low (pre-shift) -- a false alarm the world
    itself never asked for."""
    return any(e["mitigate"] and not e["blight_high"] for e in state.events)


def run_pilot() -> dict:
    cfg = WorldConfig()
    results = {}
    for beta in BETA_GRID:
        policy = instrument_mixed(beta)
        outcomes = [was_fooled(run(cfg, policy, seed=s)) for s in PILOT_SEEDS]
        results[beta] = outcomes
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("no_way_home/results/pilot_beta_sweep_v1.md"))
    args = parser.parse_args()

    results = run_pilot()

    lines = []
    lines.append("# No Way Home — PILOT Beta-Sweep (non-confirmatory)\n")
    lines.append(f"**Not a confirmatory result.** {N_PILOT_SEEDS} seeds/arm "
                 f"({PILOT_SEEDS[0]}-{PILOT_SEEDS[-1]}), used only to estimate per-arm "
                 f"variance for a power calculation. These seeds must never be reused as "
                 f"confirmatory data once Increment 3's real N is locked.\n")
    lines.append("`instrument_mixed(beta)` run directly as a policy (no mandate/election "
                 "gating), matching provenance_test_v1.md's C3b/C3c comparison structure. "
                 "\"Fooled\" = mitigated at least once during the pre-shift period "
                 "(provenance_test_v1.md's exact definition).\n")

    lines.append("| beta | Fooled / N | F(beta) |")
    lines.append("|---:|---:|---:|")
    groups_in_beta_order = []
    for beta in BETA_GRID:
        outcomes = results[beta]
        n_fooled = sum(outcomes)
        groups_in_beta_order.append([1.0 if o else 0.0 for o in outcomes])
        lines.append(f"| {beta} | {n_fooled} / {len(outcomes)} | {n_fooled/len(outcomes):.2f} |")

    import numpy as np
    rng = np.random.default_rng(42)
    J, p_value = jonckheere_terpstra_test(groups_in_beta_order, alternative="decreasing",
                                           n_permutations=9999, rng=rng)
    lines.append(f"\n## Pilot JT trend test (informative only, NOT a confirmatory result)\n")
    lines.append(f"J = {J:.1f}, one-sided permutation p-value (decreasing trend) = {p_value:.4f}, "
                 f"n_permutations=9999.\n")
    lines.append("This is a pilot with only 20 seeds/arm — not powered to be decisive either way. "
                 "Its purpose is the per-arm F(beta) values above, which feed the power calculation "
                 "for picking a real N (see run_power_calculation.py).")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
