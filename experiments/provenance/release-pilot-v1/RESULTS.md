# Provenance controls: exploratory release pilot

Equal calibration budgets; primary loss is whole-run shortfall. Historical beta reports are preserved.

Selected thresholds: raw=1, unique=0.8, decay=1.

| Scenario | Policy | Mean shortfall [bootstrap 95% CI] | False alarms | Missed crisis | Mean delay |
|---|---|---:|---:|---:|---:|
| default | raw | 38304 [37774, 38892] | 0.0% | 0.0% | 17.0 |
| default | unique | 37877 [37500, 38251] | 0.0% | 0.0% | 18.7 |
| default | decay | 35389 [34855, 35973] | 0.0% | 0.0% | 17.3 |
| default | never | 112524 [112304, 112787] | 0.0% | 100.0% | 601.0 |
| forwarding_shift | raw | 40405 [40097, 40755] | 0.0% | 0.0% | 12.5 |
| forwarding_shift | unique | 40018 [39737, 40310] | 0.0% | 0.0% | 18.6 |
| forwarding_shift | decay | 40656 [40283, 41065] | 0.0% | 0.0% | 11.8 |
| forwarding_shift | never | 112540 [112310, 112788] | 0.0% | 100.0% | 601.0 |
| reporting_and_timing_shift | raw | 69302 [68244, 70397] | 0.0% | 0.0% | 69.4 |
| reporting_and_timing_shift | unique | 63609 [62178, 65031] | 0.0% | 0.0% | 115.1 |
| reporting_and_timing_shift | decay | 71085 [70398, 71796] | 0.0% | 0.0% | 117.5 |
| reporting_and_timing_shift | never | 74713 [74493, 74949] | 0.0% | 100.0% | 401.0 |

Intervals are exploratory, per comparison; they are not multiplicity-adjusted hypothesis tests.
Never mitigate demonstrates why zero false alarms alone is insufficient. See paired differences in summary.json.

Full configs, source hashes, grids and seeds: metadata.json. Raw evaluation histories: evaluation-traces.jsonl.gz.
