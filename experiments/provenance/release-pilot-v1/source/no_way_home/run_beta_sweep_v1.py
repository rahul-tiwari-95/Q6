"""Paired reanalysis of the historical Increment 3 beta sweep.

The original locked protocol and as-run report are preserved. This driver
now permutes labels within seed blocks and writes a separate artifact.
The monotonic trend is structurally implied by the pre-shift score design;
the observed curve describes threshold crossings in this generator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from no_way_home.institutions import instrument_mixed, zero_intelligence_constrained
from no_way_home.stats import jonckheere_terpstra_test
from no_way_home.world import WorldConfig, run

BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CONFIRMATORY_SEEDS = list(range(20000, 20050))  # 50 seeds/arm, disjoint from the pilot's 9000-9019


def was_fooled(state) -> bool:
    return any(e["mitigate"] and not e["blight_high"] for e in state.events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/provenance/beta-paired-reanalysis/RESULTS.md"))
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.out}")
    cfg = WorldConfig()
    outcomes_by_beta = {}
    for beta in BETA_GRID:
        policy = instrument_mixed(beta)
        outcomes_by_beta[beta] = [was_fooled(run(cfg, policy, seed=s)) for s in CONFIRMATORY_SEEDS]

    zi_outcomes = [was_fooled(run(cfg, zero_intelligence_constrained, seed=s)) for s in CONFIRMATORY_SEEDS]

    groups_in_beta_order = [[1.0 if o else 0.0 for o in outcomes_by_beta[b]] for b in BETA_GRID]
    rng = np.random.default_rng(2026)
    J, p_value = jonckheere_terpstra_test(groups_in_beta_order, alternative="decreasing",
                                           n_permutations=9999, rng=rng, paired=True)

    f_beta = {b: sum(outcomes_by_beta[b]) / len(outcomes_by_beta[b]) for b in BETA_GRID}
    knee_betas = [b for b in BETA_GRID if b < 1.0 and f_beta[b] <= 0.10]
    knee = min(knee_betas) if knee_betas else None

    lines = []
    lines.append("# No Way Home — Historical Beta-Sweep: paired reanalysis\n")
    lines.append(f"Reuses the historical instrument, grid and seeds; corrects the permutation design: N={len(CONFIRMATORY_SEEDS)} "
                 f"seeds/arm ({CONFIRMATORY_SEEDS[0]}-{CONFIRMATORY_SEEDS[-1]}), same seed set "
                 f"reused across all arms, `instrument_mixed(beta)` run directly as a policy "
                 f"(window=30, threshold=0.30, both reused verbatim). \"Fooled\" = mitigated at "
                 f"least once during the pre-shift period.\n")

    lines.append("| beta | Fooled / N | F(beta) |")
    lines.append("|---:|---:|---:|")
    for b in BETA_GRID:
        n_fooled = sum(outcomes_by_beta[b])
        lines.append(f"| {b} | {n_fooled} / {len(CONFIRMATORY_SEEDS)} | {f_beta[b]:.2f} |")

    lines.append(f"\n## Primary test\n")
    lines.append(f"Within-seed Jonckheere-Terpstra permutation, alternative=decreasing, n_permutations=9999: "
                 f"**J = {J:.1f}, p = {p_value:.5f}**.")
    reject = p_value < 0.05
    lines.append(f"\n{'**Reject H0** (p < 0.05): a monotone decreasing trend across the beta grid is supported.' if reject else '**Fail to reject H0** (p >= 0.05): no monotone trend detected at this N.'}")

    lines.append(f"\n## Falsifiable shape claim\n")
    if knee is not None:
        lines.append(f"**Observed grid point**: the smallest tested beta with F(beta) <= 0.10 is beta={knee} "
                     f"(F({knee})={f_beta[knee]:.2f}), which is < 1.0 — partial lineage-weighting "
                     f"suffices in these tested worlds; full unique-origin dedup (beta=1.0) is not required for most "
                     f"of the forward-storm resistance.")
    else:
        lines.append(f"**No qualifying partial-weight grid point**: no beta < 1.0 reaches observed F(beta) <= 0.10 in this grid "
                     f"— F(1.0)={f_beta[1.0]:.2f}. This does not establish an all-or-nothing mechanism.")

    lines.append(f"\n## Control arm: E_zero_intelligence_constrained\n")
    zi_rate = sum(zi_outcomes) / len(zi_outcomes)
    lines.append(f"Fooled {sum(zi_outcomes)}/{len(zi_outcomes)} ({zi_rate:.2f}) on the same seeds. "
                 f"Has no beta-knob and no rate threshold, so it cannot itself produce a "
                 f"dose-response curve — reported for reference, not as a beta-grid point.")

    lines.append("\nThe decreasing trend is structurally supplied: raw >= unique, and pre-shift mitigation cannot change messages. The curve locates threshold crossings for one generator, not an emergent provenance mechanism. p=0.0001 is the Monte Carlo resolution floor, not an exact probability.")
    lines.append(f"\n## Scope\n")
    lines.append("The original equal-threshold comparison does not isolate the value of lineage information from action conservatism. A separate exploratory release pilot calibrates raw, unique and bounded-decay controllers with equal search budgets and reports whole-task outcomes; see docs/experiments/provenance-protocol.md. Historical as-run reports are unchanged.")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    args.out.with_suffix(".json").write_text(json.dumps({
        "status": "paired_reanalysis", "config": asdict(cfg),
        "seeds": CONFIRMATORY_SEEDS, "betas": BETA_GRID,
        "outcomes_by_beta": outcomes_by_beta, "zi_outcomes": zi_outcomes,
        "test": {"paired": True, "alternative": "decreasing", "permutations": 9999,
                 "rng_seed": 2026, "J": J, "p": p_value},
    }, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
