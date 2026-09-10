# Adaptation pilot protocol v1

This protocol is written before running the first pilot. The pilot is exploratory;
three seeds and a small evaluation panel cannot establish a general result.

**Question:** after learning a collection task in world A, adapting to a changed
action mapping in B, then returning to A, how much collection performance transfers,
is lost, and recovers? This is a test of adaptation and retention, not yet a test of
nested networks or a claim that a new learning method improves either.

## World and intervention

The default world is a 5×5 connected grid with four walls, one pellet, one agent,
and a 32-step finite horizon. Maps, spawns, and pellet locations vary across episodes.
Static walls, remaining pellets, and the agent position are independent state layers.
The observation includes all three layers, remaining time, and the action mapping as
a 4×4 one-hot matrix. The mapping is **observed**: this is a context-switching task,
not hidden-rule inference. There is no Hunter in this first experiment.

Action labels are 0,1,2,3. Physical directions are up, down, left, right. World A maps
labels to `(0,1,2,3)`; B maps them to `(3,2,0,1)` (a clockwise rotation of directions).
Only that mapping changes. Evaluation maps use exactly the same seeds in A and B.

The base reward is −0.01 per step plus 1 on collecting a pellet. A small
potential-based shaping term is added: `0.2 × (0.97 Φ(next) − Φ(current))`, where
`Φ = −shortest_distance_to_pellet / grid_size`. Terminal and finite-horizon timeout
potentials are zero. Shaped and base returns are both reported; **success rate is
the primary descriptive outcome**. Timeouts end the objective and are not bootstrapped.

## Baselines and state semantics

One compact Double DQN trains on A for each seed. At that boundary it is cloned:

- **Frozen after A:** no further gradient updates, replay writes, exploration or
  RNG consumption. It is evaluated on both worlds at all later boundaries.
- **Continued learning:** trains on B, then A. It retains network weights, target
  network, Adam state, replay buffer, and exploration RNG across both switches.
  Maps reset at boundaries. Exploration falls from 1 to 0.1 across the first 70%
  of A training and stays at 0.1; it is not reset at switches.

These are controls, not equal-compute competing algorithms. The frozen condition
spends no training compute after A. The continued condition gets a fixed number of
additional transitions. Replay intentionally mixes old and new worlds; the mapping
in each stored observation distinguishes their dynamics.

Network: MLP 128→64→4 on full flattened observations, Adam 0.001, gamma 0.97,
batch 64, replay capacity 12,000, warmup 256 transitions, one update per four
transitions, Double DQN targets, Huber loss, gradient clip 5, target interpolation
0.01. CPU and one Torch thread. Initialization, exploration, map generation and
evaluation have independent deterministic seed streams.

## Budget and evaluation

Default pilot: seeds 0,1,2; 12,000 training transitions per A/B/A phase, hence
108,000 total training transitions. A 300-second wall-clock admission budget is
checked between training episodes and seeds; the current episode and boundary
evaluation can finish beyond that budget. An unfinished study must be reported
as incomplete. Evaluation cost is reported separately. A stopped phase has no
fabricated boundary result.

Evaluate greedy policies on fixed held-out map seeds 900000..900015 (16 episodes
per world) before training and after each complete phase. Training reset seeds use
disjoint ranges below 900000. Evaluation neither writes replay nor changes model,
optimizer or training RNG. Every individual episode, success, length, shaped and
base return is retained. The first evaluation map supplies each illustrative
trajectory; examples are not selected for favorable outcomes.

Report per-seed success on A and B at each boundary; averages and min/max across
seeds are descriptive only. Adaptation is B(after B) − B(after A). Retention change
is A(after B) − A(after A); recovery is A(after return) − A(after B). Include the
untrained greedy baseline. More seeds, confidence intervals, a recurrent-memory
baseline and an explicit memory-reset ablation are **not included** in this milestone.
Before interpreting retention, the mean after-A success across seeds must be at
least **70%** on the fixed A panel. This is a pragmatic feasibility gate, not a
statistical significance test; per-seed values must also be reported. If it is
not met, label the pilot **baseline_underlearned** and do not call later changes
catastrophic forgetting. Passing this gate is not success on the research question.
Do not tune and rerun the pilot without recording a new protocol/version and
keeping the original result.

## Artifacts and command

`python3 -m q6.adaptation --output experiments/adaptation/pilot_v1 --dashboard dashboard/data/adaptation.json --seeds 0,1,2`

The output stores this protocol's hash and actual configuration before training,
raw training/evaluation CSVs, boundary parameter snapshots, summary JSON, illustrative
trajectories, package/runtime details and source hashes. The command refuses a
nonempty output directory. The dashboard is a derived copy, never the sole record.
