# Documentation

The [root README](../README.md) is the entry point for installing and running Q6.

## Current experiments

- [Independent bank-replication protocol](experiments/bank_replication_protocol_v1.md): three new fixed-budget exploratory banks, each paired with an equal-size uniform subset, with unchanged DDQN and prospective final-policy evaluation.
- [Frozen-policy panel protocol](experiments/panel_evaluation_protocol_v1.md), [result](experiments/panel_evaluation_results_v1.md) and [validation](validation/panel-evaluation-v1.md): uniform support improves efficient success on all eight prospective panel means (+15.30 points pooled), while success reverses on one. No training or collection; next is independent support-bank replication.
- [Equal-size bank protocol](experiments/equal_support_protocol_v1.md), [result](experiments/equal_support_results_v1.md) and [validation](validation/equal-support-v1.md): primary fresh success is close, while uniform support improves route efficiency. Identical collected weights score differently across panels; the subsequent frozen-policy study measures robustness across eight new panels.
- [Experience-coverage protocol](experiments/coverage_protocol_v1.md), [result](experiments/coverage_results_v1.md) and [validation](validation/coverage-v1.md): exhaustive DDQN reaches 80.21% fresh success versus 68.23% from collected unique states, despite better familiar-start success in the collected arm. The subsequent equal-size study separates row count from combined composition.
- [Fixed-data target protocol](experiments/fixed_targets_protocol_v1.md), [result](experiments/fixed_targets_results_v1.md) and [validation](validation/fixed-targets-v1.md): exact and Double DQN targets both pass fresh success under matched coverage; DDQN improves efficient success, but both miss the stricter efficiency and every-seed fit diagnostics. The later coverage study tests that procedure under collected current-state support.
- [Exact-target supervision protocol](experiments/supervised_protocol_v1.md) and [result](experiments/supervised_results_v1.md): the unchanged network reaches 85.42% fresh-layout success; every seed passes the supervised fresh gate, but full training fit and efficient routes remain unresolved. This is privileged supervision, not online-RL competence.
- [Supervised validation](validation/supervised-v1.md): tests, independently recomputed labels/predictions, dataset and sampling audits, and dashboard coverage.
- [A-only competence protocol](experiments/competence_protocol_v1.md) and [result](experiments/competence_results_v1.md): both fixed training sets reach 100% greedy success for every seed; fresh-map competence remains unmet.
- [Competence validation](validation/competence-v1.md): software, artifacts, and dashboard checks.
- [Prior adaptation protocol](experiments/adaptation_protocol_v2.md): the one follow-up with ten times v1's training budget and a fresh evaluation panel.
- [Adaptation v2 report](experiments/adaptation_pilot_v2.md), [raw results](../experiments/adaptation/pilot_v2/results.json), and [diagnostics](../experiments/adaptation/pilot_v2/diagnostics.json): 1.08 million transitions, 31.25% initial A evaluation success, and a failed 70% competence gate. The post-hoc random/planner checks diagnose feasibility; they do not establish forgetting or recovery.
- [Provenance protocol](experiments/provenance-protocol.md): independently calibrated counting controls and held-out scenarios.
- [Provenance errata](experiments/provenance-errata.md): limits and corrections to earlier results.
- [Roadmap](roadmap.md): what the current experiments need to show before increasing scope.
- [Contributing](../CONTRIBUTING.md): test lanes, experiment template, and reproduction records.

## History and interpretation

- [September 2026 review](../research_review/2026-09-05/README.md) separates implemented behavior, observed results, and untested explanations.
- [Archived adaptation pilot v1](experiments/adaptation_pilot_v1.md) and [its original protocol](experiments/adaptation_protocol_v1.md) preserve the initial 108,000-transition result. V2 is the final tuning run in this milestone.
- [Versions](../versions/README.md) and [articles](../articles/) record the original Krishna–Hunter self-play experiments, including failures.
- [No Way Home results index](../no_way_home/results/README.md) preserves the earlier synthetic-world findings. Consult the current errata before quoting them.
- [Original Q6 narrative](../Q6.md), [No Way Home design](../Q6%20No%20Way%20Home.md), and [archive](../archive/README.md) are historical context, not current specifications.
- [Longer-memory notes](../longer_memory/README.md) are an old backlog. The current dashboard uses static files, not the Flask/PostgreSQL system proposed there.

## Tools

The dashboard is served from this checkout with `python3 -m http.server 8080 --bind 127.0.0.1`; open `/dashboard/lab.html` for current results and `/dashboard/index.html` for historical summaries. [`dashboard/scan.py`](../dashboard/scan.py) indexes local legacy runs. The [orchestrator](../orchestrator/README.md) supervises local training processes using their documented output and resume conventions.
