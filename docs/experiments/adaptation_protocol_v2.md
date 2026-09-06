# Adaptation feasibility follow-up protocol v2

Written locally **before** the v2 run, after inspecting pilot v1. This is a
predeclared exploratory follow-up, not an externally registered confirmatory study.
The archived v1 result, source, and original protocol remain unchanged.

## Reason for this one follow-up

V1 completed 108,000 training transitions in 15.40 seconds, but mean A success
after first A was only 18.75% (seeds: 25%, 25%, 6.25%). That failed the declared
70% competence gate. A retention experiment needs a competent initial policy.

This follow-up asks whether **ten times the interaction budget** is sufficient
for the same compact learner to reach that feasibility gate. There will be
**no third tuning run in this milestone** if it fails.

## What changes

- Training: 120,000 transitions per A/B/A phase, up from 12,000. Three seeds
  0,1,2 give **1,080,000 training transitions** total.
- Evaluation: a fresh fixed panel, seeds **910000..910031**, 32 episodes per
  world at each boundary. V1's seeds 900000..900015 have now been used for
  diagnostic/model-selection decisions and are not treated as fresh held-out data.
- Wall-clock admission budget: **900 seconds**, checked between training
  episodes and seeds; the current episode and boundary evaluation may finish
  beyond the budget. An incomplete phase gets no invented boundary result.

The changed evaluation panel means comparing v1 and v2 does **not** causally
isolate the effect of training budget. This is a fresh-panel feasibility check.
The epsilon schedule retains its original definition (1→0.1 over the first 70%
of A training); the larger budget therefore stretches that same schedule over
more transitions. Do not describe the difference as extra gradient steps alone.

## What remains fixed

World, model, rewards and learner settings are unchanged from the archived v1
protocol. The world is a connected 5×5 grid, four walls, one pellet, 32-step
finite horizon. Walls, pellets, and agent occupancy are separate. Remaining
time and the action mapping are observed. There is no Hunter or hidden-rule
inference. A maps action labels to up/down/left/right `(0,1,2,3)`; B rotates
directions clockwise `(3,2,0,1)`. The sequence remains A→B→A.

Base reward: −0.01 per step and +1 for collection. Shaping:
`0.2 × (0.97 Φ(next) − Φ(current))`, shortest-path potential
`Φ = −distance / grid_size`, terminal/timeout potential zero. Both returns
are logged, but success is the primary descriptive metric.

Double DQN: MLP 128→64→4 on full observation, Adam 0.001, gamma 0.97,
replay 12,000, batch 64, warmup 256, update every four steps, Huber loss,
clip 5, target interpolation 0.01; CPU, one Torch thread. Deterministic
agent-owned initialization/exploration/replay RNG and separate map streams.

After first A the policy is cloned. Frozen-after-A never learns again.
Continued learning retains weights, target, Adam, replay and RNG across B
and return-A; only the episode resets at boundaries. These are controls,
not equal-compute competitors. Evaluation is greedy, on separate environments,
without training writes or training RNG consumption. Both A/B panels are
identical across checkpoints. The first panel map supplies each illustration.

## Interpretation and stopping rule

Retain the original **mean after-A success ≥70%** feasibility gate; report all
three seed values. Below it, label `baseline_underlearned` and do not claim
forgetting, recovery or adaptation. Above it, report the A/B boundary values
and their per-seed changes descriptively; passing feasibility is not validation
of a novel architecture or a statistically established effect.

No recurrent-memory baseline, memory-reset ablation, hidden-rule inference,
new architecture, or third tuning run is included. If this second run remains
underlearned, report the failed feasibility check and a bounded diagnosis.

## Reproduction and artifacts

```bash
python3 -m q6.adaptation --output experiments/adaptation/pilot_v2 --dashboard dashboard/data/adaptation.json --seeds 0,1,2 --phase-steps 120000 --eval-episodes 32 --eval-seed-start 910000 --max-seconds 900 --protocol-file docs/experiments/adaptation_protocol_v2.md
```

The runner writes protocol/source hashes, copies the source and protocol before
learning, retains raw training/evaluation CSVs, records model parameter count and
actual training/evaluation wall times, saves inference-only boundary weights, and
exports JSON summaries and trajectories. Reproduction requires a new output path.
