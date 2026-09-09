# Exact targets on the recorded transition graph — protocol v1

Status: locally declared before main execution, not externally registered.
Restricting training-time bootstrap choices improved route efficiency with
fixed recorded data. This study holds that restricted graph fixed and replaces
moving approximate targets with exact backward values from the same logs.

## Question and arms

Primary: **logged_graph minus constrained_bootstrap greedy efficient-success
rate**, per bank, then the equal-bank mean. Does exact target supervision on
the same recorded graph improve behavior under matched experience, replay,
loss, network and action-target presentations?

Control `constrained_bootstrap` reuses nine final 30,000-update policies from
`experiments/constrained_bootstrap/pilot_v1`, banks **1, 2, 3**, seeds **0, 1, 2**.
Treatment `logged_graph` trains nine new policies from matching seed-specific
initial online/target weights. No baseline retraining or checkpoint selection.

Original collected supports remain **59,839 / 59,984 / 59,838 states** on
256 training maps **300000–300255**. Reconstruct and verify original collector
logs and deduplicated recorded-action tables, matching the constrained control
archive. Preserve masks, absent-entry sentinels, reward/successor identities,
separate terminated/truncated flags and historical occurrence counts. Repeats
have no extra loss weight. No collection, support replacement or new outcomes.

Banks share training maps, initializations and prospective panels. Report each
bank before means/ranges/signs; crossed learner/panel cells are not independent
replications. No significance, equivalence or population-wide ranking is declared.

## Exact values from recorded information only

Let `A_log(s)` be the nonempty recorded-action set of supported state `s`.
For every recorded edge, compute float64 values in ascending remaining time:

```text
Q_log(s,a) = recorded_reward(s,a)                         if recorded end
Q_log(s,a) = recorded_reward(s,a)
             + 0.97 * max_{b in A_log(s')} Q_log(s',b)  otherwise
```

Use only recorded rewards, ends and successor indices, plus recorded-state
map/remaining-time metadata for validation/order. Every nonterminal successor
must lie in the support, retain map identity, decrement remaining time by
exactly one, and have a nonempty recorded-action set. Validate end/flag/index
consistency, finite observed rewards and valid clocks; reject violations rather
than falling back to full-world outcomes or an unrestricted maximum. Terminal
edges require no successor. Absent action values remain NaN and never enter
maxima, loss or observed-action error reductions.

Freeze float64 graph targets and their float32 training casts for all banks
before fitting. Preserve full arrays, hashes, support order, edge counts,
per-clock preparation accounting and Bellman residuals. Audit with independent
recorded-graph dynamic programming, not the production solver or full-world
optimal-Q oracle. A numerical Bellman residual tolerance of **1e-12** applies
to the float64 graph; training uses the exact saved float32 casts. These are
finite-precision solutions of the restricted graph, not exact full-world Q*.

Existing exhaustive rows may resolve logged observations, and exhaustive
transitions/Q* may remain integrity and rollout diagnostics. Neither the graph
solver nor the fixed-target optimizer may access their outcome/target arrays.
Do not add values for missing actions, imitate occurrence frequencies, alter
the behavior policy or compute training labels from new rollouts.

## Matched learning and replay

Preserve fully visible World A, observed movement rule initially, 5×5/four
walls/one pellet, horizon 32, 92 observations, existing rewards/potential
shaping, gamma **0.97**, the **20,420-parameter** 92 → 128 → 64 → 4 ReLU network,
Adam **0.001**, gradient clipping **5** and soft target **tau 0.01**.
Maintain target-network updates for procedural consistency; its predictions
do not enter fixed-target training.

The loss remains mean SmoothL1 beta 1 over each state's distinct recorded
actions, then mean over the **64 sampled states**. Gather observed edges before
target/error math. Only labels change: detached saved float32 `Q_log(s,a)`
replaces the constrained DDQN label. The optimizer receives observations,
observed masks, frozen labels and sampled indices, with no successors or
network target queries. Unobserved current-action outputs remain unsupervised.

Use unchanged `SupportSampler` / `BatchSampler`, owned
**SeedSequence([learner_seed,66301])**, drawing 64 distinct support-local
positions per update, with repeats between updates. Reconstruct the control's
full 30,000-update local/global/map streams and count arrays. Treatment must
match exact ordered replay and exposure per bank/seed, not just totals.
Every supported state must be sampled by the full budget, with zero direct
off-support samples. No cross-bank replay identity is assumed.

Nine fits × **30,000 updates** give **270,000 new updates / 17,280,000 state
presentations**. Both arms have **24,273,701 action-target presentations**:
**1,062,190 terminal plus 23,211,511 nonterminal-labelled targets**. Bank
totals are **8,115,408 / 8,078,271 / 8,080,022**. Match actual sampled target
counts to archived counts and reductions from the recorded masks.

**Neural successor-query counts are intentionally unequal:** the historical
control has **23,211,511** optimizer successor queries; the fixed-target
treatment has **zero**. Recorded nonterminal edge presentations are still
matched; they must not be relabelled as neural queries. Count unique graph
preparation edges/successor lookups, preparation wall time, and final-fit
diagnostic inference separately. This is an equal-presentation/update study,
not an equal-compute or equal-neural-query study. Preserve resource totals.

## Final supported-state fit diagnostics

After all nine fits, measure every final policy on every supported state of
its own bank, including states with no successful path in the restricted
graph. Use deterministic gradient-free online inference in bounded batches
(at most **1,024 states**); no target-network inference is needed here.
Save dense float32 K×4 predictions in support order, final snapshot identities
and unchanged parameter/RNG checks. Both arms receive the same diagnostic
inputs and float64 graph references. This is **18 final fits assessed**, with
**1,077,966 state presentations / 1,513,974 observed-edge errors** in the main
run; these are diagnostic counts, not optimizer targets or policy episodes.

Report per bank/condition/seed, with denominators explicit:

- State-balanced MAE/MSE: mean observed-action error within each state, then
  mean across supported states. Squared error precedes both means.
- Edge-pooled MAE and maximum absolute observed-action error, separately.
- Restricted action agreement: select lowest-label online argmax over logged
  actions; agreement if its float64 graph value is within **1e-6 absolute,
  zero relative tolerance** of the maximum logged graph value.
- Graph regret: mean of maximum logged graph value minus the value of that
  selected logged action. Include all supported states.
- Fraction of states whose unrestricted online argmax falls outside their
  logged-action set. Do not assign missing graph values to those choices.

Report graph-reference fit separately from full-world rollout Q* diagnostics;
they have different targets and sampled states. Final fit is descriptive, with
no new gate, model selection, extra fitting or fixed-point claim for a neural
approximation. No new checkpoint probe study or full checkpoint fit sweeps.
Full audits regenerate saved final-fit predictions; portable audits recompute
metrics from saved predictions and independently checked graph labels.

## Evaluation and interpretation

Freeze inputs, supports, controls, exact labels and all prospective panels
before training. Train sequentially by bank then seed. Preserve **45 treatment
snapshots** at 0/1,000/3,000/10,000/30,000 updates, **nine copied control
finals**, and **2,700 loss windows**. All treatment fits precede final-fit
diagnostics and every policy rollout. Evaluate only final policies, without
gradients/replay/optimizer; check source/copy/online/target identities around
every model-panel evaluation. No subsequent tuning or second main run.

Use eight new disjoint 64-map panels, scanning from **1120000 + 1000 × panel**.
Exclude training, named supervised/fixed-target/coverage/equal-support fresh
panels, and all selected layouts from frozen-panel, bank-replication,
map-replay, within-map, recorded-action and constrained-bootstrap studies.
Exclude already accepted layouts. Identity is walls + pellet + rules, ignoring
seed/spawn/clock; retain original spawns and every rejection. No selection by
difficulty/outcome. This does not exclude every historical Q6 world.

Fresh greedy actions remain unrestricted over all four predictions; use the
same epsilon-0.1 diagnostic with paired external draws
**[learner_seed,map_seed,repetition,55219]**, lowest-label ties. Each final
model/map has one greedy and two epsilon episodes. **27,648 learner episodes**
comprise 9,216 greedy and 18,432 epsilon. Shared references add **3,072 random
plus 512 planner episodes**, totaling **31,232** on 512 accepted layouts.

Efficient success means success within twice the planner shortest-path length,
with failures included. Each bank/condition has 1,536 greedy episodes. Preserve
4,608 greedy layout pairs, 81 seed pairs (72 panel plus nine all-panel), and
27 aggregate pairs (24 panel plus three all-panel); no invented epsilon pairs.
Report success, efficient success, all-episode steps and blocked steps.
Equal-count bank means equal episode pooling for success/efficiency/steps;
pooled blocked ratios, mean bank ratios and mean learner ratios remain distinct.
Successful-only averages compare different successful subsets.

Historical 70% success/above-random and 80% efficient-success crossings remain
descriptive references, not new competence gates. Eligibility means completed
and internally consistent evidence, not performance success. Preserve all
earlier gates and results. Older policies outside these two arms are not
ranked using scores from different panels.

This comparison holds the recorded action graph and restricted Bellman
operator fixed while testing moving approximate targets against exact fixed
labels. Exact graph values need not yield better fresh-world behavior; the
graph excludes unrecorded actions and unobserved current-action predictions
remain free. A remaining fit/generalization gap cannot automatically be
assigned solely to optimization or coverage. These are offline recorded-state
experiments, not online competence, adaptation/retention or a memory result.

## Dashboard, resources and reproducibility

Add **Exact logged graph** to the existing dashboard, showing frozen constrained
controls, matched data/replay/action-target counts, intentionally unequal
neural queries, graph preparation, final graph-fit metrics and common-panel
behavior. Do not invent control training curves or reuse old bootstrap/probe
diagnostics as new observations. Preserve earlier studies and initially visible
rules. Preselect first accepted map per panel, repetition zero, both modes and
all bank/arm/seed combinations: **288 learner + 16 shared reference = 304
recordings**, with no favorable substitution.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU,
one Torch thread and `nice -n 10`. Admission cap **1,200 seconds** includes
preparation, learning, graph-fit inference, evaluation and aggregation. Sampled
peak-process-RSS guard **4 GiB** is not a hard OS/total-machine limit. Guard,
nonfinite or identity failures preserve partial evidence and disqualify the run.
Review/push this protocol, then freeze clean pushed source before one main run.

Capture source/protocol/git/environment/command; manifests/logs/tables;
supports/labels/casts/graph preparation; controls/initials/snapshots;
sampling/counts/target access; fit predictions/metrics/inference accounting;
panels/rejections; losses/raw episodes/pairs/replays/resources; SHA-256 manifest.
Output `experiments/logged_graph/pilot_v1`, dashboard
`dashboard/data/logged_graph.json`. License remains undecided.

Smoke uses all three original supports, seed 0 and 24 updates per bank:
**72 updates**, six new snapshots plus three archived 30,000-update finals.
Use two two-map panels starting **1200000 / 1201000**, producing **72 learner
+ 12 reference episodes**, **28 recordings**, and six final supported-state
fit assessments. Match the short treatment replay/target stream to the
historical first-24-update prefix while retaining full control budget labels.
Unequal smoke budgets are ineligible scientific evidence. Tests may use
smaller fixtures. Audit without retraining or full policy reevaluation.
