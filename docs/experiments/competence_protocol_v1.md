# A-only competence diagnosis — protocol v1

Status: locally declared before this study's training, not externally registered.
Prior evidence: both A→B→A feasibility pilots failed their initial competence gate.
Their results and evaluation panels have already informed this diagnostic. This
is a separate milestone, not a retroactive extension of their protocols.

## Question and hypothesis

Can the existing compact Double DQN reliably learn a small repeated collection
task before we ask it to learn across a stream of different tasks?

The working hypothesis is that the learner can acquire a useful greedy policy
on a small repeated support, but its learning setup struggles with many layouts.
Compare one repeated task, sixteen repeated tasks, and a task stream. This is a
feasibility diagnosis of the complete learning setup, not a proof that network
capacity, exploration, bootstrapping, or generalization is the sole cause.

## Fixed design

- World A only: existing `WorldConfig` defaults, 5×5, four walls, one pellet,
  32-step finite horizon, normal controls, visible walls/pellet/position/clock
  and action mapping. Rewards, shaping, discounts and collision rules unchanged.
  A task instance fixes walls, initial agent position and initial pellet.
- Learner: current 20,420-parameter Double DQN, hidden layers 128/64, Adam 0.001,
  gamma 0.97, replay 12,000, batch 64, warmup 256, update every four transitions,
  target interpolation 0.01, gradient norm cap 5. No action masking, planner
  demonstrations, new representation, reward change, or architecture search.
- Independent learner initializations: seeds 0, 1, 2. All three conditions use
  the same initial policy for a given seed. World/task sampling, exploration,
  replay, and evaluation have recorded RNG ownership. The existing learner's
  exploration and replay share its owned RNG, as in the previous pilots.
- Each condition/seed receives exactly 120,000 training transitions. Epsilon
  falls linearly from 1 to 0.1 over the first 84,000 transitions and remains 0.1.
  Evaluate at 0, 4,000, 12,000, 40,000 and 120,000 transitions. Continue the same
  episode and learner state across evaluation checkpoints.

| Condition | Training task selection | Training probe panel |
|---|---|---|
| `fixed_1` | Repeat task seed 200000 | That one task |
| `fixed_16` | Seeds 200000–200015, shuffled balanced epochs | All sixteen tasks |
| `stream` | Those sixteen tasks once initially; then independent uniform draws from 0–899999, excluding those sixteen IDs | The initial sixteen tasks |

Task IDs were selected without inspecting their layouts or outcomes for this
milestone. Streaming draws may repeat; record both distinct IDs and distinct
canonical task layouts. Balanced episodes do not balance transitions: difficult
tasks occupy more steps. Preserve per-task episode and transition counts. The
stream's probes were encountered once initially; they are not a repeated-support
memorization test. No probe has been encountered at the zero-step checkpoint.

## Evaluation and references

The common fresh-map panel is **920000–920063**. These IDs are outside the training
range and exclude the previous 900000/910000 evaluation panels. Distinct IDs are
not a guarantee of distinct layouts; record canonical task hashes and any exact
training/evaluation task overlap. No model or checkpoint selection uses this
panel. Report all scheduled checkpoints and the final fixed-budget outcome.

Evaluate from each task's original state under both:

1. Greedy actions: one deterministic episode per map and learner checkpoint.
2. Epsilon 0.1: eight independent rollouts per probe map and two per fresh map.
   Use independent per-map/repetition evaluation RNG. A given learner seed,
   map and repetition shares its exploration draws across conditions and
   checkpoints; no evaluation consumes training RNG or updates weights/replay.

References receive the same visible information and horizon. Uniform random
actions use seeds 0/1/2 with the same 8/2 repetition counts. A shortest-path
reference is evaluated once per map. It computes routes without learning and
is not a compute-matched neural competitor. Repeated presentation of a common
reference across conditions does not create additional independent evidence.

## Outcomes and interpretation gates

Primary outcomes are **final greedy success on each condition's probe panel**
and **final greedy success on the common fresh panel**, reported per seed.

- Repeated-support competence: `fixed_1` or `fixed_16` reaches at least **90%**
  greedy probe success for **each** of the three learner seeds. This supports
  learning those specific tasks, not general spatial reasoning.
- Permission to resume meaningful cross-world research: at least one neural
  condition reaches **70%** greedy fresh-map success for **each** seed, and
  exceeds the random reference on that panel. This is a descriptive feasibility
  gate, not a significance test; meeting it warrants independent replication.
- If fixed-support competence fails, the next diagnosis remains learning on
  known tasks. If it passes while fresh performance fails, distinguish learning
  the selected support from useful transfer; this comparison alone does not
  identify whether coverage, representation or optimization causes the gap.
- If epsilon evaluation helps, it identifies sensitivity to action selection
  on these maps. It does not by itself show that insufficient training
  exploration caused the original failure.

Secondary diagnostics: base/shaped return, episode length, blocked-action
fraction, moved-position revisits (excluding no-ops), and mean absolute learned
Q error against an exact finite-horizon reference along evaluation trajectories.
The independent dynamic-programming reference uses only visible state and the
declared rules. Record selected-action optimal regret and the fraction of
optimal actions **among states where completion is still possible**; report
that denominator so already unwinnable states do not inflate optimality.
Q comparisons use shaped values consistent with the training target. The oracle
is evaluation-only and never supplies replay transitions, targets or actions
during learning. These are observations of errors, not causal explanations.

Curves show means and ranges over learner seeds, not confidence intervals.
The common task bank limits generality: three initializations do not constitute
three independent task banks. Preserve every outcome, including negative ones.

## Budget, reproducibility and stopping rule

Maximum main-study training: **1,080,000 transitions** (three conditions × three
seeds × 120,000). One CPU thread, no GPU. Wall-clock admission cap: **900 seconds**
including evaluation; if reached, save a clearly incomplete result and do not
present partial coverage as a passed gate. Environment setup and software tests
are outside this training budget. No automatic extra seeds or tuning run.

Use Python 3.12 with Torch 2.8.0 and NumPy 2.0.2 for the main run; record the full
resolved environment and platform. Earlier pilots used Python 3.9, so comparing
their scores with this stream arm is not an isolated causal comparison. Within
this study, all arms use the same runtime. Smoke runs must be labeled as such.

Before training, save the protocol, resolved configuration, source snapshots,
hashes, Git revision/dirty state, seeds and command. Preserve training and
evaluation CSVs, seed-level summaries, inference-only checkpoint weights,
first-map replays selected independently of outcomes, source/runtime metadata
and a SHA-256 artifact manifest. The dashboard consumes those saved results.

Stop after the declared comparison, inspect the gates and diagnostic evidence,
and write a separate result report. Any following experiment requires a new
explicit hypothesis and budget. Historical pilots remain unchanged. A separate
post-hoc analysis of their saved weights may be published with that label.
