# Prior v2: exact-Q diagnosis of frozen after-A policies

Post-hoc, read-only analysis of saved v2 weights. No training or exploration was performed.
The same32 A evaluation maps and three saved training seeds reproduce all96 original episodes exactly.

| Training seed | Success | Blocked steps | Optimal actions while winnable | Mean optimal-Q regret while winnable | Mean absolute Q error |
|---:|---:|---:|---:|---:|---:|
| 0 | 31.25% | 74.21% | 16.12% | 0.0630 | 0.5033 |
| 1 | 34.38% | 73.10% | 17.02% | 0.0617 | 0.5808 |
| 2 | 28.12% | 72.57% | 15.94% | 0.0641 | 0.6923 |

The reference computes exact optimal discounted shaped Q from visible walls, pellet, position, remaining time and action mapping. Its independent finite-horizon dynamics are checked against actual world transitions. It is an engineered diagnostic, not a learned or compute-matched competitor.

These policies exhibit incorrect greedy action rankings and many blocked/revisited positions. Learned values also underestimate optimal values on average. This identifies observed behavior, not why learning produced it: optimization, state coverage, representation and exploration have not been causally separated.

Step metrics are trajectory-weighted: a32-step failure contributes more observations than a short success. Once the remaining deadline makes success impossible, optimal actions can tie; ranking fraction/regret are therefore also restricted to still-winnable states. Absolute Q error averages all four actions over visited states. No significance or independence claim is attached to these correlated steps.

This historical panel already informed diagnosis and is not a fresh test of the new A-only experiment. The source snapshots, exact old model hashes and measured CPU runtime are in results.json/source/manifest.json. Original pilot files were not changed.

Reproduce from repository root with a new output directory:

```bash
python3 experiments/competence/prior_v2_diagnostic/run.py --out /tmp/q6-prior-v2-reproduction
```
