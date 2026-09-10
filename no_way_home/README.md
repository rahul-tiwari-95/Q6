# No Way Home

A small synthetic decision laboratory: four locality sickness counts, shared food/medicine/wealth, copied outbreak reports, a hidden regime shift and one collective mitigation action. It is an inspectable aggregate stochastic-control model; the broader multi-agent society remains an earlier design target.

## Current release (2026-09-06)

The [bounded provenance pilot](../docs/experiments/provenance-results.md) compares equally calibrated raw-count, unique-origin-count and bounded-decay controllers on fresh seeds and two parameter shifts. A decayed raw counter wins on default shortfall; unique counting performs better under the selected shifts. No universal winner or novel learned provenance mechanism is claimed.

**Read the [errata](../docs/experiments/provenance-errata.md) before citing older results.** The historical beta monotonicity is supplied by the score/outcome design, and its permutation analysis originally ignored seed pairing. The learner receives a deduplicated feature. Those original reports remain unchanged as records of what was run. New code fixes run-local message IDs and terminal learner updates; new artifacts preserve source snapshots and raw evaluation histories.

```bash
python3 -m pytest tests/test_no_way_home_smoke.py tests/test_stats.py tests/test_nwh_release.py -q
python3 -m no_way_home.run_provenance_release --out experiments/provenance/my-reproduction
python3 -m no_way_home.run_beta_sweep_v1 --out experiments/provenance/my-beta-reanalysis.md
```

Run directories/reanalysis reports must be new. The provenance pilot uses NumPy and runs504,000 ticks without training; existing stats tests also require SciPy. Full checks include the older, slower learner regressions.

## Where to read next

- [Pilot protocol](../docs/experiments/provenance-protocol.md), [results and interpretation](../docs/experiments/provenance-results.md), [complete machine-readable artifact](../experiments/provenance/release-pilot-v1/results.json).
- [Historical results narrative](results/README.md), including non-degenerate economy fixes, report lineage and the mandate-commitment ablation.
- [Earlier specification](PREREGISTRATION.md), [locked historical beta protocol](PREREGISTRATION_INCREMENT3.md), [environment redesign](ENVIRONMENT_REDESIGN.md), and [historical phase plan](ENGINEERING_NWH_PHASE_PLAN.md).
- `world.py` (kernel), `messages.py` (report/forward lineage), `policies.py` (scripts), `institutions.py` (supplied voting/mandate rules), `learning.py` (tabular action learner), `stats.py` (independent or seed-paired permutation), `run_provenance_release.py` (bounded calibration/evaluation).

No new neural architecture, corroboration mechanism or institutional layer is needed to reproduce this release.
