"""CONFIRMATORY beta-sweep for Increment 3. Executes exactly what
PREREGISTRATION_INCREMENT3.md locks -- grid, instrument, N, seed range, and
test -- committed before this script was ever run against these seeds. If
anything here diverges from that file, the pre-registration file is the
one that's wrong and needs fixing, not this script silently drifting from it.

Also runs the E_zero_intelligence_constrained control arm on the same seeds
(ready with no further calibration). The bounded-decay-EMA control arm is
NOT run here -- its threshold calibration is a separate, not-yet-done step
per PREREGISTRATION_INCREMENT3.md, and this result should not wait on it.

Usage: python3 -m no_way_home.run_beta_sweep_v1
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--out", type=Path, default=Path("no_way_home/results/beta_sweep_v1.md"))
    args = parser.parse_args()

    cfg = WorldConfig()
    outcomes_by_beta = {}
    for beta in BETA_GRID:
        policy = instrument_mixed(beta)
        outcomes_by_beta[beta] = [was_fooled(run(cfg, policy, seed=s)) for s in CONFIRMATORY_SEEDS]

    zi_outcomes = [was_fooled(run(cfg, zero_intelligence_constrained, seed=s)) for s in CONFIRMATORY_SEEDS]

    groups_in_beta_order = [[1.0 if o else 0.0 for o in outcomes_by_beta[b]] for b in BETA_GRID]
    rng = np.random.default_rng(2026)
    J, p_value = jonckheere_terpstra_test(groups_in_beta_order, alternative="decreasing",
                                           n_permutations=9999, rng=rng)

    f_beta = {b: sum(outcomes_by_beta[b]) / len(outcomes_by_beta[b]) for b in BETA_GRID}
    knee_betas = [b for b in BETA_GRID if b < 1.0 and f_beta[b] <= 0.10]
    knee = min(knee_betas) if knee_betas else None

    lines = []
    lines.append("# No Way Home — Beta-Sweep, Increment 3 (CONFIRMATORY)\n")
    lines.append(f"Executes `PREREGISTRATION_INCREMENT3.md` exactly: N={len(CONFIRMATORY_SEEDS)} "
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
    lines.append(f"Jonckheere-Terpstra, alternative=decreasing, n_permutations=9999: "
                 f"**J = {J:.1f}, p = {p_value:.5f}**.")
    reject = p_value < 0.05
    lines.append(f"\n{'**Reject H0** (p < 0.05): a monotone decreasing trend across the beta grid is supported.' if reject else '**Fail to reject H0** (p >= 0.05): no monotone trend detected at this N.'}")

    lines.append(f"\n## Falsifiable shape claim\n")
    if knee is not None:
        lines.append(f"**Knee exists**: the smallest beta with F(beta) <= 0.10 is beta={knee} "
                     f"(F({knee})={f_beta[knee]:.2f}), which is < 1.0 — partial lineage-weighting "
                     f"suffices, full unique-origin dedup (beta=1.0) is not required to get most "
                     f"of the forward-storm resistance.")
    else:
        lines.append(f"**No partial credit**: no beta < 1.0 reaches F(beta) <= 0.10 in this grid "
                     f"— F(1.0)={f_beta[1.0]:.2f}. Dedup looks all-or-nothing at this resolution.")

    lines.append(f"\n## Control arm: E_zero_intelligence_constrained\n")
    zi_rate = sum(zi_outcomes) / len(zi_outcomes)
    lines.append(f"Fooled {sum(zi_outcomes)}/{len(zi_outcomes)} ({zi_rate:.2f}) on the same seeds. "
                 f"Has no beta-knob and no rate threshold, so it cannot itself produce a "
                 f"dose-response curve — reported for reference, not as a beta-grid point.")

    lines.append(f"\n## Not yet in this result\n")
    lines.append(f"The bounded-decay-EMA control arm (`messages.py::bounded_decay_rate`) still needs "
                 f"its threshold calibrated on held-out seeds before it can run as a fair comparison "
                 f"— see `PREREGISTRATION_INCREMENT3.md`. This is a real, named gap in the "
                 f"confound-ruling-out story, not a silent omission: the result above establishes "
                 f"the dose-response shape, not yet that it's specifically about lineage rather than "
                 f"any bounded/decayed counting.")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
