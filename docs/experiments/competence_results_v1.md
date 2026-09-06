# A-only competence diagnosis — result v1

**The current learner solves the selected repeated tasks, but does not establish
useful performance on fresh worlds.** After 120,000 training transitions per
condition and learner seed, both the one-task and sixteen-task conditions reached
100% greedy probe success for all three seeds. No condition passed the fresh-world
gate. The working hypothesis received support on this task bank; the cause of
weak transfer remains unresolved.

The [protocol](competence_protocol_v1.md) was locally declared before training,
not externally registered. The full comparison completed without protocol
deviations: 1,080,000 training transitions, 45 scheduled checkpoints, and
160.16 seconds including evaluation, within the 900-second admission cap.
No additional training run or checkpoint selection followed these results.

## Final fixed-budget outcomes

World A exposes its controls from the start. The world, rewards, shaping,
20,420-parameter Double DQN, and training budget are identical across conditions.
The fixed task bank is shared across learner initializations 0, 1, and 2.
Each fresh-world greedy score uses the same 64 maps, seeds 920000–920063.

| Training condition | Greedy probe success, seeds 0 / 1 / 2 | Greedy fresh success, seeds 0 / 1 / 2 | Mean fresh success | Repeated-support gate | Fresh-world gate |
| --- | --- | --- | ---: | --- | --- |
| One repeated task | 100 / 100 / 100% | 9.38 / 10.94 / 9.38% | 9.90% | Passed | Not met |
| Sixteen repeated tasks | 100 / 100 / 100% | 31.25 / 18.75 / 17.19% | 22.40% | Passed | Not met |
| Task stream | 12.50 / 37.50 / 31.25% | 23.44 / 23.44 / 12.50% | 19.79% | Not applicable | Not met |

Repeated-support competence requires at least 90% greedy success for **every**
seed. Fresh-world competence requires at least 70% for **every** seed and scores
above the common random reference. Stream probes are the sixteen tasks shown
once at the start, not a repeatedly trained memorization panel.

The one-task condition reached 100% probe success for all seeds by the first
4,000-step checkpoint. The sixteen-task condition did so by 40,000 steps.
Both remained perfect at the final checkpoint. Their final greedy routes on
the probes were shortest paths, with no blocked actions or revisits.

Uniform random actions achieved **42.45%** fresh-map success over 384 rollouts
(64 maps × three RNG seeds × two repetitions). The shortest-path reference
solved **100%** of those maps in one rollout each. Reference results are reused
across conditions; they are not three independent reference datasets. The planner
uses the visible rules without learning and is not a compute-matched neural
competitor. These descriptive scores do not establish a population effect.

## What action selection reveals

| Condition | Fresh success, greedy | Fresh success, epsilon 0.1 | Greedy blocked steps | Optimal actions while still winnable | Mean absolute Q error |
| --- | ---: | ---: | ---: | ---: | ---: |
| One task | 9.90% | 15.36% | 89.02% | 5.40% of 5,092 states | 0.3566 |
| Sixteen tasks | 22.40% | 30.21% | 67.03% | 17.59% of 4,679 states | 0.1332 |
| Stream | 19.79% | 29.95% | 76.19% | 13.48% of 4,762 states | 0.6335 |

Epsilon evaluation uses two rollouts per fresh map and eight per probe map,
with independent evaluation randomness and unchanged weights. Its better fresh
scores show sensitivity to action selection on these maps. They do not show
that training exploration alone caused the failure. All gates remain greedy.

Q error compares all four learned action values with an independent exact
finite-horizon dynamic-programming reference, using the same shaped-return
definition as training. This reference only observes evaluation; it supplies
no training actions, targets, or transitions. Fresh greedy action regret is
0.0598 / 0.0584 / 0.0592 for the three conditions. Q error, regret, and blocked
fractions are weighted by visited steps, so long failed episodes contribute
more. Optimal-action fractions exclude states where success is already
impossible. These metrics describe failure trajectories; they do not isolate
representation, coverage, optimization, or bootstrapping as the cause.

The [separate post-hoc analysis of pilot v2](../../experiments/competence/prior_v2_diagnostic/RESULTS.md)
reproduces its saved after-A outcomes and finds similar blocked behavior.
It is historical diagnosis, not another training comparison.

## Exposure and limits

| Condition | Distinct task IDs, seeds 0 / 1 / 2 | Distinct physical tasks, seeds 0 / 1 / 2 | Episode starts, seeds 0 / 1 / 2 |
| --- | --- | --- | --- |
| One task | 1 / 1 / 1 | 1 / 1 / 1 | 37,264 / 37,414 / 37,126 |
| Sixteen tasks | 16 / 16 / 16 | 16 / 16 / 16 | 21,564 / 21,518 / 21,529 |
| Stream | 5,904 / 5,870 / 5,599 | 5,901 / 5,863 / 5,598 | 5,932 / 5,884 / 5,613 |

Every row received 120,000 transitions per seed. Stream task IDs are drawn with
replacement after the first sixteen; different IDs can generate the same
physical task. Canonical hashes found **no exact training/fresh-task overlap**
in any condition/seed. This checks full task identity, not every form of shared
structure. Balanced fixed-bank episodes do not balance transition exposure.

Three learner seeds share one selected fixed bank and one fresh panel. Their
ranges are not confidence intervals or independent task-bank replications.
Success on the fixed bank establishes learning those particular instances,
not general navigation. Earlier adaptation pilots ran on Python 3.9 with
different protocols and panels; their scores cannot isolate the effect of
this study's change in sampling or runtime.

## Artifacts and reproduction

The main run used clean source revision
`88fb9d227ecfad12cda7699e74781bfd1d1bf46b`, Python 3.12.13, Torch 2.8.0,
NumPy 2.0.2, one CPU thread on macOS arm64. Training occupied 121.52 seconds,
learner evaluation 36.52 seconds, and unique references 1.69 seconds. These are
observed times on this machine, not a throughput benchmark.

The [artifact directory](../../experiments/competence/pilot_v1/) contains the
[results](../../experiments/competence/pilot_v1/results.json), compressed training
episodes, evaluation and reference CSVs, per-task exposure, 192 recorded
trajectories, 45 inference-only checkpoints, seven source snapshots, resolved
environment, protocol, command, and a [62-file SHA-256 manifest](../../experiments/competence/pilot_v1/manifest.json).
Checkpoints are for inspection, not resumable training.

```bash
# In the Python 3.12 development environment; use a new output directory.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
python -m q6.competence --output experiments/competence/my-reproduction \
  --protocol-file docs/experiments/competence_protocol_v1.md
python scripts/verify_pilot_artifacts.py
```

For the complete environment, consult the captured `environment.txt`; library
and hardware differences can affect training. The [validation record](../validation/competence-v1.md)
separates software checks from this scientific result.

## Decision after this milestone

Keep A→B→A and memory extensions paused: general initial competence remains
below the declared gate. The next useful question is whether the existing
representation can learn navigation across tasks when supplied reliable targets.
A bounded supervised fit to the exact reference would separate that diagnostic
from online exploration and bootstrapped targets. Strong fresh performance there
would justify focusing next on the RL learning procedure; weak performance would
motivate a controlled spatial-representation comparison. Neither outcome alone
would identify a single cause, and a supervised result would not be an RL result.
That follow-up needs its own hypothesis, untouched evaluation panel and budget;
it has not been run here.
