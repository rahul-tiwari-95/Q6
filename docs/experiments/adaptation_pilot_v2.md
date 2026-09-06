# Adaptation feasibility v2: stop at the baseline gap

**The larger run still did not establish a competent initial policy.** Mean
held-out A success after first A was **31.25%**, below the locally predeclared 70%
gate. The same-panel post-hoc random-action reference reached **41.67%**, and a
shortest-path reference reached **100%**. No third training run was performed.

This is a completed, reproducible feasibility check with an underlearned neural
baseline. It does not support claims about catastrophic forgetting, recovery,
novel architectures, or efficient adaptation. The lab now makes that limitation
visible instead of disguising low competence as an interesting retention curve.

## Predeclared run results

Three seeds, with 32 fixed held-out maps per world. Values below are mean greedy
success rates. “Untrained” is a greedy randomly initialized network, not the
random-action reference introduced later.

| Boundary | Continued: A | Continued: B | Frozen after A: A | Frozen after A: B |
|---|---:|---:|---:|---:|
| Untrained | 9.38% | 5.21% | — | — |
| After first A | 31.25% | 16.67% | 31.25% | 16.67% |
| After B | 9.38% | 25.00% | 31.25% | 16.67% |
| After return to A | 14.58% | 6.25% | 31.25% | 16.67% |

The frozen policy's parameter hash and per-map outcomes remain identical across
boundaries. The continued policy changes, but initial competence is too low to
interpret the changes as the desired adaptation/retention experiment.

V2 used **1,080,000 training transitions**, **36,399 evaluation transitions**, and
a **20,420-parameter** MLP. Recorded CPU time was **132.88 seconds training**,
**1.64 seconds evaluation**, and **134.88 seconds total** on this machine. These
are local timings, not a comparative simulator benchmark.

Training budget increased tenfold from [v1](adaptation_pilot_v1.md), with the same
world/network/reward/learner definitions. The epsilon schedule stretches with that
budget. V1's evaluation panel had already informed the follow-up, so v2 uses fresh
seeds 910000..910031. The changed panel means the two outcomes do not causally
isolate a training-budget effect. The [v2 protocol](adaptation_protocol_v2.md) was
written locally before this run; it was not externally registered.

## Separate post-hoc references

These diagnostics were requested after inspecting the underlearned run. They are
saved separately and are **not retroactively included as predeclared outcomes**.
Neither reference trains. Both act through the same one-step environment API and
the same 32-step horizon.

| Reference | A success | B success | Evaluation count per world |
|---|---:|---:|---:|
| Shortest path from visible observation | 100% | 100% | 32 maps, once each |
| Seeded uniform random actions | 41.67% | 46.88% | 32 maps × 3 RNG seeds |

The planner parses visible walls, remaining pellets, agent position, and the
visible action-mapping matrix. It neither reads hidden state nor bypasses walls,
timeouts, or movement rules. Its mean successful route is 3.53 steps in both
worlds. This establishes solvability of these panels under the exposed rules;
it is an engineered reference, not a learned or compute-matched competitor.

## Diagnosis and next gate

| Seed | Last 100 completed A training episodes | Greedy held-out A after first A |
|---|---:|---:|
| 0 | 49% | 31.25% |
| 1 | 47% | 34.38% |
| 2 | 29% | 28.13% |

Training performance itself is weak. The issue is therefore not explained solely
by successful optimization followed by failure to generalize to new maps.
Training and evaluation also differ in exploration (training still uses ε=0.1),
so their gap does **not** isolate generalization. Small TD losses are not evidence
that the policy is competent.

A review found no analogous wrong-action replay or invented-successor issue in
this new learner. Targeted tests independently check Double DQN action selection,
terminal masking, an actual parameter update, exact optimizer/replay/RNG
continuation, observation-visible mapping, finite-horizon reward consistency,
and isolated evaluation. Tests reduce specific implementation risks; they do not
prove the entire learning setup is correct or the chosen hyperparameters adequate.

The next gate should be **A-only competence**, before any memory architecture or
another A→B→A study: inspect greedy no-op/loop behavior, compare a fixed-map task
with fresh-map evaluation under the same exploration setting, and verify a small
baseline against the shortest-path reference. That is a proposed future diagnostic,
not a third run hidden in this milestone. The causes of weak neural learning have
not yet been isolated.

Post-pilot API review found that a custom replay capacity below the fixed
256-transition warmup would silently prevent learning. The current constructor
now rejects that configuration and nonpositive batch/observation sizes. Both
pilots used capacity 12,000 and batch 64, so this guard does not explain or change
their results. Their archived source and results remain unchanged; no retraining
was performed for the guard.

## Inspect or reproduce

- [Actual protocol and source hashes](../../experiments/adaptation/pilot_v2/protocol.json)
- [Raw training episodes](../../experiments/adaptation/pilot_v2/training.csv)
- [Raw evaluation episodes](../../experiments/adaptation/pilot_v2/evaluations.csv)
- [Summary, per-seed results and trajectories](../../experiments/adaptation/pilot_v2/results.json)
- [Post-hoc references and training-tail diagnostics](../../experiments/adaptation/pilot_v2/diagnostics.json)
- [Reference episodes](../../experiments/adaptation/pilot_v2/diagnostics_episodes.csv)

Exact pre-run source snapshots are in each pilot's `source/q6/`; their bytes were
verified against that pilot's original protocol hashes. V1 protocol/results were
not rewritten. Boundary weight files are labeled inference-only, not resumable.

```bash
python3 -m q6.adaptation --output experiments/adaptation/reproduction_v2 --seeds 0,1,2 --phase-steps 120000 --eval-episodes 32 --eval-seed-start 910000 --max-seconds 900 --protocol-file docs/experiments/adaptation_protocol_v2.md
python3 -m q6.diagnostics --study experiments/adaptation/reproduction_v2
```
