# Documentation

The [root README](../README.md) is the entry point for installing and running Q6.

## Current experiments

- [Exact logged-graph protocol](experiments/logged_graph_protocol_v1.md), [result](experiments/logged_graph_results_v1.md) and [validation](validation/logged-graph-v1.md): exact recorded-graph targets improve value MAE 0.2473 → 0.1134 but reduce fresh efficient success 48.70% → 18.64%, with declines in all 24 bank-panel means. Next is a frozen-policy familiar-start diagnostic with unrestricted versus logged-mask actions.
- [Constrained-bootstrap protocol](experiments/constrained_bootstrap_protocol_v1.md), [result](experiments/constrained_bootstrap_results_v1.md) and [validation](validation/constrained-bootstrap-v1.md): logged successor choices improve efficient success 22.66% → 49.52% with identical data, replay and target counts; all 24 bank-panel means improve in efficiency. The subsequent exact-target study uses the same recorded graph.
- [Recorded-action protocol](experiments/recorded_actions_protocol_v1.md), [result](experiments/recorded_actions_results_v1.md) and [validation](validation/recorded-actions-v1.md): exact state replay with logged-only action outcomes reduces efficient success 43.03% → 21.29%, with declines in all three banks; the subsequent constrained-bootstrap control uses the same recorded data.
- [Within-map composition protocol](experiments/within_map_protocol_v1.md), [result](experiments/within_map_results_v1.md) and [validation](validation/within-map-v1.md): replacing states with exact map quotas and original replay schedules improves efficiency in all three banks (+19.47/+28.65/+20.70 points); the subsequent recorded-action study uses the original collected supports.
- [Equal-map replay protocol](experiments/map_replay_protocol_v1.md), [result](experiments/map_replay_results_v1.md) and [validation](validation/map-replay-v1.md): balancing map exposure works, but efficiency effects are mixed (−3.91/+7.16/+4.56 points); the subsequent within-map study preserves exact map quotas and original replay schedules.
- [Independent bank-replication protocol](experiments/bank_replication_protocol_v1.md), [result](experiments/bank_replication_results_v1.md) and [validation](validation/bank-replication-v1.md): three new matched-size pairs replicate the uniform efficiency advantage (+17.12/+27.67/+23.31 points); the subsequent replay study tests equal map exposure on the same collected supports.
- [Frozen-policy panel protocol](experiments/panel_evaluation_protocol_v1.md), [result](experiments/panel_evaluation_results_v1.md) and [validation](validation/panel-evaluation-v1.md): uniform support improves efficient success on all eight prospective panel means (+15.30 points pooled), while success reverses on one. No training or collection; the subsequent bank study tests independent support draws.
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
