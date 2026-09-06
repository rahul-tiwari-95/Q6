# Frozen-policy panel robustness — protocol v1

Status: locally declared before main execution, not externally registered.
The equal-size comparison found a uniform-bank efficiency advantage, but the
identical collected network scored 68.2% on the previous fresh panel and 83.9%
on the next. This study measures panel variation before changing learning.

## Question and fixed policies

Does the **uniform-minus-collected efficient-success advantage** repeat across
eight new 64-map panels? Report task success and episode length separately;
include exhaustive training as context. This is evaluation of existing policies,
not new learning or independent support-bank replication.

Load only final **30,000-update** snapshots, learner seeds **0, 1, 2**, in order:

1. `collected_unique` from `experiments/equal_support/pilot_v1`.
2. `uniform_subset` from `experiments/equal_support/pilot_v1`.
3. `exhaustive` from `experiments/coverage/pilot_v1`.

All nine policies retain the same 20,420-parameter 92 → 128 → 64 → 4 ReLU
network. Use inference-only loading, evaluation mode and disabled gradients;
allocate no optimizer or replay buffer. Online weights choose actions. Archive
target weights for provenance but never use them to choose evaluation actions.
Validate online/target tensor hashes and snapshot file hashes before and after
evaluation, including per-panel checks. A mismatch makes the run ineligible.
No updates, support draws, training-state collection, refits or model selection.

Keep fully visible World A: 5×5, four walls, one pellet, horizon 32, controls
visible initially, 92-value observation, discount 0.97, unchanged rewards and
potential shaping. Preserve the preceding rollout and external exploration
draw procedures. These policies were trained with privileged four-action
offline transitions; this study does not establish online-RL competence.

## Prospective panels

Select `panel_0` through `panel_7` in that order, scanning upward from
**970000 + 1000 × panel index**, accepting **64 layouts** per panel. Reject
identities matching any of the 256 training layouts (300000–300255), any layout
in the four preceding fresh panels (supervised 930000, fixed-target 940000,
coverage 950000, equal-size 960000 scans), or any layout already accepted in
this study. Identity is walls + pellet + rules, ignoring spawn, clock and seed.
Use archived accepted layout lists for exclusions, not assumed contiguous IDs.
Retain the generator's original spawn, candidate order and skip reasons.

Finish all panel selection before learner predictions or outcomes. Do not
balance, replace or select maps using distance, difficulty, policy values or
success. Record layout identities, selected seeds and pairwise zero overlap.
The panels are disjoint prospective generator samples; policies and support
banks are shared. Eight panels × three learner seeds are not 24 independently
trained policy/bank replications.

## Evaluations and references

For every model/map evaluate one greedy episode and two epsilon-0.1 episodes.
Use existing exploration draws keyed by `[learner_seed, map_seed, repetition,
55219]`, identical across conditions. Greedy ties choose the lowest action
label. No panel-specific or condition-specific draw enters this key.

Shared references: random seeds 0/1/2, two repetitions per map, plus one
deterministic shortest-path planner episode per map. Compute references on
these selected maps only. Exact finite-horizon Q* along evaluated trajectories
may measure action quality; it never guides a learned policy. Do not enumerate
the full training/fresh state banks or repeat full-state fit evaluation.

Main counts: **512 maps**, **13,824 learner episodes** (4,608 greedy and 9,216
epsilon), **3,584 reference episodes** (3,072 random and 512 planner),
**17,408 total evaluation episodes**. Each condition has 1,536 greedy and 3,072
epsilon episodes; each panel/condition has 192 greedy and 384 epsilon episodes.
Evaluation steps are measurements, not training examples or collection cost.

## Aggregation and interpretation

Primary robustness contrast: **uniform_subset minus collected_unique greedy
efficient-success rate**, where an episode succeeds in at most twice its own
map's planner length. Include failures in the denominator. Report every panel
and learner seed, all-panel per-seed totals, and all-model-seed pooled totals.
All panels receive equal weight; equal map/repetition counts make this equal
to episode pooling. Retain additive counts to make denominators auditable.

Also report success, all-episode and successful-only lengths, pooled blocked
steps / total steps, returns and existing rollout action-quality diagnostics.
A mean of per-seed blocked fractions is distinct from the pooled fraction;
label these explicitly and do not substitute one for the other.

Save paired greedy per-layout/seed differences in success, efficient success,
steps and no-op fraction for all three ordered contrasts:
`uniform_subset − collected_unique`, `exhaustive − collected_unique`, and
`exhaustive − uniform_subset`. Save panel/seed, panel-pooled, all-panel/seed and
overall summaries. Expected paired rows: 4,608 per-layout, 72 panel/seed,
24 panel-pooled, nine all-panel/seed and three overall contrasts.

Describe panel min/max ranges and positive/zero/negative panel counts for each
paired contrast, emphasizing the primary efficiency contrast. Show previous
70% success and 80% efficiency reference lines, and comparisons with each
panel's pooled random rate, only as descriptive threshold crossings. **No new
competence gate**, unchanged old gates, no full-state training-fit gate here.
Do not choose a different model, checkpoint, action mode or panel subset after
seeing results. No significance, population-generalization or equivalence claim
follows from these repeated fixed policies. Exhaustive results contextualize
the sparse-bank comparison without identifying a particular composition cause.

Archive the first accepted map per panel, repetition zero, all nine models in
both modes, plus random seed 0 and planner: **160 replays**. Selection precedes
outcomes; individual examples need not resemble pooled results. Show panel
comparisons and selectable traces in the existing dashboard, preserving older
studies. Memory and online feedback remain deferred.

## Resources, provenance and stopping

Pinned main runtime: Python 3.12, Torch 2.8.0, NumPy 2.0.2, deterministic CPU,
one Torch thread, sequential work at reduced scheduler priority (`nice -n 10`).
Admission cap **1,200 seconds** includes selection, loading, rollout and
aggregation. Sample platform-correct process peak RSS; stop admitting work
above **4 GiB**. This is a sampled process guard, not an OS allocation limit or
total-machine measurement. Preserve partial measurements and stop reasons.

Commit/push this protocol and independently review it before execution. Freeze
clean source before the single main run. Capture git/source/protocol hashes,
runtime, command, selected layouts/skips, input archive metadata/file hashes,
nine copied inference snapshots, before/after model checks, raw learner and
reference rows, paired tables, summaries, replays, resource timings and a
SHA-256 manifest. Main eligibility requires complete expected cells/counts,
unchanged models, no training/collection and all resource/provenance checks.
Eligibility is evidence integrity, not a competence claim.

Output `experiments/panel_evaluation/pilot_v1` and export directly to
`dashboard/data/panel_evaluation.json`. Preserve previous captured sources and
artifacts. Keep license undecided.

Smoke: two panels of two maps, starts **1010000** and **1011000**, learner seed
0 in all three conditions, loading the same archived final policies. This is
36 learner episodes + 12 references = 48 total, with 16 preselected replays.
No smoke training or altered weights. Smoke/partial/time/memory/hash failures
are ineligible main evidence. Do not add panels, retrain or retune after the
main run. Audit saved raw rows and selected replay forward passes without a
second complete learned-policy evaluation; portable CI may skip those forward
passes while preserving artifact and aggregate validation.
