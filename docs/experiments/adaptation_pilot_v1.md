# Adaptation pilot v1: baseline underlearned

The run completed, but **does not yet provide evidence about forgetting or recovery**.
Mean held-out A success after the first training phase was **18.75%**, below the
70% feasibility gate written in the [protocol](adaptation_protocol_v1.md) before
the run. The per-seed values were 25%, 25%, and 6.25%.

This is an exploratory feasibility result for the new lab. It is not an improvement
claim, a failed test of nested networks, or a demonstration that the task is inherently
hard. The current compact baseline did not learn the initial task reliably within
the allocated budget.

## Observations

Greedy success on the same 16 held-out maps per world, averaged over three seeds:

| Boundary | Continued: A | Continued: B | Frozen after A: A | Frozen after A: B |
|---|---:|---:|---:|---:|
| Before training | 8.33% | 6.25% | — | — |
| After first A | 18.75% | 4.17% | 18.75% | 4.17% |
| After B | 20.83% | 16.67% | 18.75% | 4.17% |
| After return to A | 22.92% | 12.50% | 18.75% | 4.17% |

The “before training” row is the untrained greedy control, not a random-action
policy. The frozen control retains exactly the after-A policy and performs no more
training. These are descriptive proportions on a small panel, not confidence bounds
or a statistical test. Because initial competence was insufficient, the boundary
differences must not be described as established adaptation, forgetting, or recovery.

## Budget and checks

The run used seeds 0, 1, 2 and 12,000 transitions per A/B/A phase: **108,000 training
transitions**, plus **19,300 evaluation transitions**. Reported wall time was 15.40
seconds on this machine's CPU, including 14.11 seconds in training and 0.92 seconds
in evaluation. These local timings are not a cross-platform performance benchmark.

Fourteen targeted tests passed. They check connected objectives, independent state
layers, visible rule changes, finite-horizon behavior, exact world/RNG continuation,
potential-reward consistency, evaluation isolation, learning-state restoration, and
a short end-to-end study. Raw evaluation rows preserve policy hashes; the frozen
policy stays unchanged across all boundaries. Illustrations use the first fixed
evaluation map, without selecting favorable trajectories.

## Reproduce and inspect

- [Actual protocol and source hashes](../../experiments/adaptation/pilot_v1/protocol.json)
- [Raw evaluation episodes](../../experiments/adaptation/pilot_v1/evaluations.csv)
- [Raw training episodes](../../experiments/adaptation/pilot_v1/training.csv)
- [Summary and per-seed results](../../experiments/adaptation/pilot_v1/results.json)
- [Illustrative trajectories](../../experiments/adaptation/pilot_v1/trajectories.json)

Run from the repository root, using a new output directory:

```bash
python3 -m q6.adaptation --output experiments/adaptation/reproduction_v1 --seeds 0,1,2
```

The inference snapshots in `models/` contain boundary network weights. They are
explicitly labeled **not resumable**; full replay/optimizer continuation is supported
by the in-memory agent API and tested, but is not claimed for these smaller files.

The next experimental gate is reliable held-out A competence. A follow-up should
change one declared factor, retain this result, and stop short of a larger adaptation
study until that gate is met. Recurrent memory and memory-reset comparisons remain
outside this milestone.
