# Exact-target supervision — protocol v1

Status: locally declared before this study's main training, not externally
registered. The preceding [A-only diagnosis](competence_results_v1.md) informed
this experiment: the existing network learned its selected fixed tasks but
failed fresh-world competence. Its previously observed evaluation panels are
not reused for this study's primary outcome.

## Question and hypothesis

Can the **same network and observations** learn useful navigation across layouts
when supplied exact action-value targets and a fixed, broad state dataset?

The working hypothesis is that reliable targets and explicit state coverage
allow this representation to learn useful fresh-layout behavior. This is an
oracle-supervised capacity/fitting diagnostic, not an RL benchmark or a causal
ablation of target quality alone. It changes both the source of targets and
state coverage relative to online RL. A failure cannot uniquely identify
network capacity, optimization, or generalization as its cause.

## Fixed task, model and training procedure

- World A is unchanged: 5×5 grid, four walls, one pellet, 32-step finite
  horizon, ordinary controls visible from the beginning. The observation
  contains the same wall, pellet and position layers, remaining-time scalar,
  and action-mapping matrix (92 values). Existing rewards, discount 0.97 and
  potential shaping are unchanged.
- Use `DQN.online` exactly as implemented: 92 → 128 → 64 → 4 with ReLU,
  **20,420 parameters**, independent initialization seeds **0, 1, 2**. Start
  each seed from its normal untrained initialization, not a previous checkpoint.
- Reuse Adam at **0.001**, batch size **64**, gradient norm cap **5**. Minimize
  PyTorch Smooth L1 loss, beta 1, averaged over the batch and all four action
  values. Targets are fixed exact shaped finite-horizon Q* values from
  `VisibleOptimalQ`, converted to float32 for training. No target-network
  updates, replay transitions, TD bootstrapping, exploration collection,
  action masking, auxiliary loss, architecture change, or hyperparameter search.
- Run **30,000 optimizer updates per seed**, evaluated at updates
  **0, 1,000, 3,000, 10,000, 30,000**. Minibatches draw 64 distinct row indices
  uniformly from the fixed training dataset; batches can repeat rows. Each
  seed owns an RNG initialized by `SeedSequence([seed, 66301])`. Evaluation
  does not consume it or alter parameters/optimizer state.
- The budget is **1,920,000 supervised state presentations per seed**,
  **5,760,000 total**, each supplying four privileged action targets. This
  is approximately 11.72 dataset-equivalent passes per seed, not shuffled
  epochs and not 5.76 million environment interactions. Preserve sample
  counts and a digest of sampled batch indices.

## Dataset and split

The shared training bank uses task seeds **300000–300255**. For each map's
walls and pellet, enumerate every non-wall, non-pellet agent position in
row-major order and every remaining-time value **1–32**, in increasing order.
There are 20 positions × 32 times = **640 states per bank entry**, or
**163,840 training states**. Preserve the chosen ordering in metadata.

These are hypothetical valid Markov states. Some cannot be reached from the
original sampled start at the corresponding elapsed time. Some cannot finish
before the deadline. Keep them for Q regression, and distinguish them in
evaluation. The finite-horizon return ends at success or timeout. Terminal
observations do not require a decision and are not training rows.

Identify a layout by **walls + pellet + declared rules**, ignoring the original
agent position, clock, and RNG seed. Disjoint task IDs alone are insufficient:
training covers all positions. Keep any duplicate training-bank entries and
report them explicitly; they receive the corresponding repeated weight.

Select **64 fresh layouts** by scanning seeds upward from **930000**, rejecting
any layout identical to a training layout or an already accepted fresh layout.
Record every accepted seed, rejected seed and reason. Selection uses layout
identity only, before predictions or outcomes; do not reject difficult maps.
Enumerate the same 640 states on every fresh layout (**40,960 states**).
No fresh label, state, rollout, or score can affect an optimizer update or
checkpoint choice. All seeds share this bank and panel.

Save the observation/target arrays, layout IDs, row metadata, exact target
generation code and hashes. Record and check zero train/fresh layout and
observation intersections. A split check does not rule out shared local
structure or guarantee generalization beyond this generator.

## Evaluations and references

At every scheduled checkpoint, evaluate:

1. **Original-start episodes:** all 256 training-bank starts and all 64 fresh
   starts, using one greedy rollout and two epsilon-0.1 rollouts per map.
   Greedy uses the lowest action label on a predicted-value tie. Epsilon draws
   use the existing independent evaluation RNG keyed by learner seed, map seed,
   repetition and 55219; common draws are reused across checkpoints and policies.
2. **Full-state predictions:** every enumerated training and fresh state, with
   batched inference and no updates. Report per seed, map and remaining-time
   bucket **1–8, 9–16, 17–24, 25–32**, as well as the pooled panel.

Report rollout success, return, length, blocked actions, moved-position revisits
and action-value diagnostics. References use the same visible information and
horizon. Uniform random actions use seeds 0/1/2 with two repetitions per map;
shortest path uses one deterministic rollout per map. Shared reference records
do not create additional independent evidence.

Also evaluate the three **frozen final stream policies** from competence v1
on the same fresh panel under greedy and epsilon-0.1 actions. They retain their
original 120,000-transition training history; no RL training is added. Preserve
their source/model hashes. Audit wall-and-goal overlap with their recorded
training tasks and report it per seed; fresh selection is not conditioned on
this historical overlap. “Fresh” means unseen by the supervised training bank,
not automatically by these older policies. They are historical references,
**not matched for data, target access, state coverage or compute**. Scores from
the old 920000 panel are not plotted as if measured on this panel.

## Metrics and decision criteria

The primary outcome is **final greedy success on the 64 fresh layouts, per
learner seed**. A useful supervised policy requires **at least 70% in every
seed and above the pooled random reference**. Passing warrants replication
under this supervised setting; it does not satisfy the online-RL competence
requirement for resuming A → B → A.

The training-fit diagnostic requires **at least 90% greedy success on the
entire training bank and 90% optimal-action agreement among winnable training
states, in every seed**. This is an operational diagnostic, not proof of exact
function fitting. Do not invent a universal cutoff for Q error.

Full-state metrics include four-action mean absolute error, RMSE, signed bias,
95th percentile of per-state mean absolute error, chosen-action regret
`max Q* − Q*(chosen)`, and optimal-action agreement. Label an action optimal
when its exact value is within absolute **1e-6**, relative tolerance zero,
of the best exact value. Preserve denominators and separate winnable from
impossible states: impossible states can give trivial optimal-action ties.
Report winnable-state regret separately, and original-start rollout metrics
separately from the uniformly enumerated-state metrics. Low Q error alone
does not establish useful action ordering or episode success.

Interpret outcomes without selecting a better intermediate checkpoint:

- Strong final training and fresh behavior supports useful policy capacity
  under exact supervision. Focus the following diagnosis on the online RL
  procedure, while recognizing that this comparison also changed coverage.
- Strong training behavior with weak fresh behavior supports a transfer
  limitation in this setting. A controlled coverage/representation comparison
  becomes reasonable; it does not prove the current representation is incapable.
- Weak training behavior leaves fitting, optimization, capacity and sampling
  unresolved. Diagnose that fit before attributing failure to online RL alone.

Curves show means and ranges over three initializations, not confidence
intervals. Shared task banks are not independent task-bank replications.

## Budget, provenance and stopping rule

One main run: at most **90,000 optimizer updates**, one CPU thread, **900 seconds**
including dataset/reference preparation, training and evaluation. Save a clearly
incomplete result if the admission cap is reached; partial checkpoints cannot
pass either gate. No automatic extra training, seeds or tuning run.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2** and record the full resolved
runtime. Before main training, capture this protocol, configuration, command,
Git revision/dirty state, sources and hashes. Retain per-100-update loss records,
sample exposure counts/digests, raw rollout and state metrics, every scheduled
inference-only checkpoint, final dense predictions, and first-map replays
chosen independently of outcomes. Save an artifact SHA-256 manifest and export
the dashboard directly from saved results. Smoke/tests are separate from the
main budget and must not be presented as evidence that a research gate passed.

Stop after the fixed comparison. Preserve its outcome and write a separate
results report. Any following experiment needs a new hypothesis and budget.
Historical artifacts and the owner's undecided license remain unchanged.
