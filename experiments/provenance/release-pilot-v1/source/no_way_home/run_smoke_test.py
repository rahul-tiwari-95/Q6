"""Run the A1 nontriviality smoke test: every permanent control policy,
5 seeds each, and report whether the world produces separable signal on the
primary metric before any learning is built. See PREREGISTRATION.md §6.

Usage: python3 -m no_way_home.run_smoke_test
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from no_way_home.metrics import mitigation_rate, need_shortfall_per_10k
from no_way_home.policies import POLICIES
from no_way_home.world import WorldConfig, run

DEFAULT_SEEDS = list(range(10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-localities", type=int, default=4)
    parser.add_argument("--n-agents-per-locality", type=int, default=6)
    parser.add_argument("--n-ticks", type=int, default=2000)
    parser.add_argument("--shift-tick", type=int, default=800)
    parser.add_argument("--out", type=Path, default=Path("no_way_home/results/smoke_test_v1.md"))
    args = parser.parse_args()

    cfg = WorldConfig(n_localities=args.n_localities, n_agents_per_locality=args.n_agents_per_locality,
                       n_ticks=args.n_ticks, shift_tick=args.shift_tick)

    rows = []
    for name, policy_fn in POLICIES.items():
        overall_vals, pre_vals, post_vals, mit_rates = [], [], [], []
        for seed in args.seeds:
            state = run(cfg, policy_fn, seed=seed)
            overall_vals.append(need_shortfall_per_10k(state))
            pre_vals.append(need_shortfall_per_10k(state, (1, cfg.shift_tick)))
            post_vals.append(need_shortfall_per_10k(state, (cfg.shift_tick, cfg.n_ticks + 1)))
            mit_rates.append(mitigation_rate(state))
        rows.append({
            "policy": name,
            "overall_mean": statistics.mean(overall_vals),
            "overall_std": statistics.pstdev(overall_vals),
            "pre_shift_mean": statistics.mean(pre_vals),
            "post_shift_mean": statistics.mean(post_vals),
            "post_shift_std": statistics.pstdev(post_vals),
            "mitigation_rate": statistics.mean(mit_rates),
        })

    rows.sort(key=lambda r: r["overall_mean"])

    lines = []
    lines.append(f"# No Way Home — A1 Nontriviality Smoke Test\n")
    lines.append(f"Config: {cfg.n_agents} agents, {cfg.n_ticks} ticks, shift at tick "
                  f"{cfg.shift_tick}, seeds {args.seeds}.\n")
    lines.append("Primary metric: need-shortfall agent-ticks per 10,000 ticks (lower is better). "
                  "Ranked best to worst overall.\n")
    lines.append("| Policy | Overall (mean±std) | Pre-shift | Post-shift (mean±std) | Mitigation rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['overall_mean']:.1f} ± {r['overall_std']:.1f} | "
            f"{r['pre_shift_mean']:.1f} | {r['post_shift_mean']:.1f} ± {r['post_shift_std']:.1f} | "
            f"{r['mitigation_rate']:.2f} |"
        )

    # Nontriviality verdict: does the BEST policy meaningfully beat ZI-C, and
    # do policies separate at all, or does everything land in the same place?
    # Deliberately measured against the best REALIZED score, not against
    # greedy_state_oracle specifically -- see policies.py's docstring on why
    # that policy is diagnostic information, not a guaranteed upper bound.
    by_name = {r["policy"]: r for r in rows}
    best = rows[0]
    oracle = by_name["greedy_state_oracle"]
    zic = by_name["C2_zero_intelligence_constrained"]
    zi = by_name["C1_zero_intelligence"]
    need_first = by_name["C3_need_first_heuristic"]

    best_vs_zic = (zic["overall_mean"] - best["overall_mean"]) / max(zic["overall_mean"], 1e-9)
    lines.append("\n## Verdict\n")
    lines.append(f"Best policy ({best['policy']}) vs. ZI-C relative gap: {best_vs_zic:.1%} "
                 f"(best={best['overall_mean']:.1f}, ZI-C={zic['overall_mean']:.1f}).")
    if best_vs_zic < 0.10:
        lines.append("\n**STOP-CONDITION TRIGGERED**: nothing beats ZI-C by more than 10%. "
                      "The physical world barely discriminates between an informed policy and "
                      "a random-but-constrained one -- per PREREGISTRATION.md §6, this means the "
                      "world design needs to change before building anything further on top of it.")
    else:
        lines.append(f"\nThe world discriminates: the best policy beats ZI-C by {best_vs_zic:.1%}, "
                      f"well above the 10% stop-condition threshold. Real signal exists for a "
                      f"policy to capture.")
    lines.append(f"\nZI ({zi['overall_mean']:.1f}) vs. ZI-C ({zic['overall_mean']:.1f}): the budget "
                 f"constraint alone {'materially changes' if abs(zi['overall_mean']-zic['overall_mean'])/max(zic['overall_mean'],1e-9) > 0.05 else 'barely changes'} "
                 f"outcomes.")

    if need_first["overall_mean"] < oracle["overall_mean"]:
        gap = (oracle["overall_mean"] - need_first["overall_mean"]) / max(oracle["overall_mean"], 1e-9)
        lines.append(f"\n**Notable, not a bug:** the need-first heuristic ({need_first['overall_mean']:.1f}) "
                     f"beats greedy_state_oracle ({oracle['overall_mean']:.1f}) by {gap:.1%}, despite having "
                     f"strictly worse information (a lagging observed signal vs. the true hidden state). "
                     f"The oracle knows the truth but spends its wealth budget greedily and naively -- every "
                     f"tick blight is high, not accounting for the shared scarce budget -- and runs its "
                     f"buffer dry partway through the post-shift window. need_first only tries to mitigate "
                     f"when observed symptoms are already elevated, which happens to also be a more "
                     f"budget-conservative spending pattern. Perfect state information isn't sufficient here; "
                     f"spending discipline under a shared resource constraint is a separate skill, and this "
                     f"toy world already shows the two can trade off against each other. That's a real "
                     f"argument for why a learned policy -- which could jointly optimize inference AND "
                     f"timing -- might add something beyond either simple heuristic alone.")
    else:
        lines.append(f"\ngreedy_state_oracle ({oracle['overall_mean']:.1f}) beats need_first "
                     f"({need_first['overall_mean']:.1f}), as naively expected from strictly better "
                     f"information.")

    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
