# Frozen policies, familiar starts and action support — protocol v1

Status: locally declared before main execution, not externally registered.
Exact recorded-graph labels improved supported-state numerical fit but harmed
fresh-map behavior. Retain constrained DDQN as the learning baseline and locate
that regression before changing training again.

## Frozen crossed comparison

Reuse all **18 final 30,000-update policies** captured by
`experiments/logged_graph/pilot_v1`: `constrained_bootstrap` and `logged_graph`,
banks **1, 2, 3**, learner seeds **0, 1, 2**. No checkpoint selection, optimizer,
training, replay, new collection, label fitting or memory. Verify source, copied
snapshot and online/target parameter identities before and after evaluation.

Every policy starts at the original spawn with **32 moves remaining** on all
**256 training maps, 300000–300255**. Verify these reset observations against
archived map/spawn metadata and each bank's recorded support. Cross each policy
with two deterministic action sets:

- `unrestricted`: lowest-label argmax of all four online predictions.
- `logged`: lowest-label argmax among actions actually recorded at the current
  map/position/remaining-time state in that policy's bank.

The mask changes selection only; weights and observations are identical.
Logged rollouts must remain in closed recorded support with a nonempty action
set at every nonterminal state. Reject inconsistencies; no fallback action,
oracle outcome or guessed mask. Unrestricted rollouts may leave support.
World A remains fully visible, including the movement rule initially: existing
5×5 map, four walls, one pellet, horizon 32, 92 observations, rewards/shaping,
gamma 0.97 and 20,420-parameter feedforward architecture are unchanged.

This gives **9,216 learner episodes**, 2,304 per condition/action-set cell,
768 per bank/cell and 256 per bank/seed/cell. No epsilon episodes and no new
fresh-map panels. Divide the ascending training seeds into eight contiguous
32-map blocks for descriptive consistency checks. These are familiar blocks,
not prospective held-out panels or independent replications.

## References and attainable behavior in the logs

Reuse archived float64 `Q_log` labels and reconstruct/check the recorded tables
against original logs. Independently audit the backward recurrence. Run one
lowest-label exact recorded-Q greedy reference per bank/start: **768 episodes**.
It maximizes discounted return within the logged graph; do not assume this
automatically makes it a success ceiling.

Separately compute success reachability and shortest successful recorded path
by ascending remaining clock using only logged terminal flags and successors:
a successful terminal edge has length one; a nonterminal edge has length one
plus its successor's finite successful distance; all other edges are infinite.
The state minimum determines reachability and shortest logged success. Validate
closure, decreasing clock, flags and reference path consistency. This gives a
logged success ceiling and logged efficient-success ceiling for each start,
without inventing missing edges. Preserve the arrays and per-start results.

Run **256 shared full-world shortest-path reference episodes**, once per start,
to define the existing efficiency threshold and show how restrictive the logs
are. Full-world reference information never controls learner selection or
logged reachability. Total main episodes: **10,240**. Graph references are
bank-specific and are not duplicated across learner seeds.

## Paired outcomes and regression investigation

Primary contrast: the equal-bank mean **difference of masking effects on
efficient success**,
`(exact_logged - exact_unrestricted) - (DDQN_logged - DDQN_unrestricted)`.
Report each bank first, then mean/range/signs, retaining paired bank/seed/start
rows. Also report exact-minus-DDQN within each action set and
logged-minus-unrestricted within each model. All differences use identical
starts. Include success, efficient success (success within twice the full-world
shortest route, failures included), all-episode steps, base/shaped returns and
blocked steps. Keep pooled blocked ratios separate from means of ratios;
successful-only step averages concern different successful subsets.

Preserve per-step action values, choices, masks, support membership, clock,
position, reward and terminal flags. Distinguish:

- Off-mask choices **while the current state is supported**, with supported
  decision count as denominator; first such decision and episode incidence.
- First **nonterminal successor outside support**, number of exits/reentries,
  off-support decision occupancy and episode incidence. An unlogged action
  can still land on a supported state; terminal completion is not support exit.
- Logged-value regret of selected actions when defined, restricted action
  agreement at absolute tolerance **1e-6**, and loss of logged success
  reachability along masked trajectories. Missing graph values stay missing.

Use the prior archived full-support predictions, without new inference, to
report restricted ranking separately for states with one versus multiple
logged actions, successful-path reachability and remaining-time groups
1–8 / 9–16 / 17–24 / 25–32, plus the original-start clock-32 slice.
Preserve state-balanced signed value bias as
well as absolute/squared error. On multiple-action states, also center each
state's observed-action prediction errors by their within-state mean and
report centered MAE/MSE, separating a shared value offset from relative
recorded-action errors. Report target best-versus-second-best logged-action
gaps and predicted unrestricted-versus-best-logged value gaps without
assigning targets to missing actions. These are predeclared explanatory
slices, not fitted interventions or new performance gates. They analyze
already available prior predictions; the original-start slice is an exploratory
prior-data finding documented before the new familiar rollout outcomes, not
a confirmatory result on unseen data. One-action states have automatic
restricted agreement. Lower numerical error alone cannot establish better
behavior. Summarize paired start outcomes and any divergence
in the predetermined replays without selecting maps by results.

A larger masking rescue for the exact model would support an action-selection
mismatch involving outputs without local supervision. A residual masked gap
would locate additional error within supported choices. Familiar unrestricted
regression would show that novel maps are not required for the problem. These
controls cannot uniquely assign every fresh-map failure to one mechanism,
establish online competence, or turn bank-specific masks into a deployment
feature. Previous fresh-map results remain separate context; do not recompute
or pool them with these familiar starts. No significance claim, new gate,
additional training run or post-outcome hypothesis selection.

## Dashboard and evidence

Add **Familiar starts** to the existing dashboard with an explicit 2×2 display,
paired masking effects, graph references, support diagnostics and action-set
selection. Label masks as privileged familiar-state diagnostic access, and
keep movement rules visible initially. Preserve all **2,618 prior recordings**.
Predetermine maps **300000 + 32 × i**, i=0…7: all 18 models in both action sets
plus three graph references and one shared planner per map gives **320 new
recordings**. No favorable substitutions.

Capture protocol/source/git/environment/command, input manifests and hashes,
snapshots, recorded tables and support, starts and graph reachability,
raw episodes/step diagnostics, paired reductions, prior-prediction slices,
replays, resources and a SHA-256 manifest. Output
`experiments/familiar_starts/pilot_v1` and
`dashboard/data/familiar_starts.json`. Preserve prior artifacts byte for byte.
An independent audit checks recorded-graph DP, paired reductions and evidence
identities. Portable mode uses saved predictions; full audit can verify selected
replay predictions without repeating the complete learned-policy evaluation.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU,
one Torch thread and `nice -n 10`. Admission cap **1,200 seconds** includes
preparation, evaluation and aggregation. Sampled peak-process-RSS guard
**4 GiB** is not a hard OS or total-machine limit. Guard/nonfinite/identity
failures preserve partial evidence and disqualify the run. Review and push
this protocol, then freeze clean pushed implementation before **one main run**.
License remains undecided; PR #1 stays open.

Smoke uses banks 1–3, seed 0, original maps **300000 and 300032** in two blocks:
**24 learner + 6 graph + 2 planner = 32 episodes/recordings**. It reuses final
policies without fitting and checks execution only; it is not research evidence.
Tests may use smaller synthetic graphs. Do not repeat the main to tune results.
