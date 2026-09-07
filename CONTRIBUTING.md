# Contributing to Q6

Useful contributions make an experiment easier to reproduce or its interpretation easier to test. Start with a small issue or pull request describing the observed behavior, the proposed change, and how you checked it. Include the failing seed and command for a bug. For a larger experiment, use the template below before spending the training budget.

## Development setup

Use Python 3.10+ in a virtual environment and work from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Runtime dependency bounds live in `pyproject.toml`; `requirements.txt` and `requirements-dev.txt` are compatibility installers for this checkout. The `test` extra contains pytest and SciPy (used to check the custom statistics against a reference implementation). The `dev` extra adds build and metadata-checking tools. These are compatibility ranges, not an exact reproduction lock: preserve the resolved environment with each published experiment.

## Tests

The fast lane excludes exactly three existing training-heavy cases. It keeps environment, replay, agent-update, statistics, orchestration, and new compact experiment checks. To reproduce the seeded, single-thread CPU setup used in CI:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import random
import sys
import numpy as np
import pytest
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)
sys.exit(pytest.main([
    "tests", "-q",
    "--deselect=tests/test_train_phase2.py::test_smoke_run_writes_artifacts",
    "--deselect=tests/test_no_way_home_smoke.py::test_floored_decay_learner_reliably_avoids_forward_storm_bins",
    "--deselect=tests/test_no_way_home_smoke.py::test_floored_decay_learner_outcome_is_stable_across_training_amounts",
]))
PY
```

The full lane includes all cases, including five episodes of the legacy CNN self-play trainer and two 50-versus-200-episode tabular-learner checks:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import random
import sys
import numpy as np
import pytest
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)
sys.exit(pytest.main(["tests", "-q"]))
PY
```

GitHub Actions runs the fast lane for pushes and pull requests on Python 3.10 and 3.12. The full lane runs weekly or through **Actions → tests → Run workflow → full_suite**. The workflow describes intended checks; consult the actual run for their status. Seeding makes this CPU check repeatable within an environment, not a promise of identical results across hardware and library versions.

Run focused tests while developing. Run the relevant broader lane before submitting changes to dynamics, observations, rewards, learner updates, persistence, or statistical analysis. Add tests for a meaningful invariant or regression, not just a second copy of the implementation. Do not replace scientific evaluation with a passing smoke test.

Check the shipped pilot source/protocol hashes, CSV aggregates, training counts, supervised dataset split/predictions/gates, and artifact manifests without retraining:

```bash
python scripts/verify_pilot_artifacts.py
node scripts/check_dashboard.mjs
```

CI also runs these checks and checks the current dashboard JavaScript syntax. The dashboard harness exercises saved-data controls with DOM/canvas stubs; inspect the actual browser for layout and native interaction. Integrity does not establish independent replication or scientific validity.

For deeper read-only audits of the completed target studies, use their recorded Python 3.12, Torch 2.8.0 and NumPy 2.0.2 environment:

```bash
python scripts/audit_supervised_study.py
python scripts/audit_fixed_targets_study.py
python scripts/audit_coverage_study.py
python scripts/audit_equal_support_study.py
python scripts/audit_panel_evaluation.py
python scripts/audit_bank_replication.py
python scripts/audit_map_replay.py
```

These separately check exact targets, reconstruct minibatch sampling schedules, reproduce final dense predictions from saved weights, and reconcile raw metrics and gates. They perform forward inference and isolated environment transitions, with no training or policy rollouts. Bit-exact predictions require the recorded environment and are excluded from broader dependency-range CI. The supervised audit accepts complete, undeviating v1 studies; its smoke and incomplete artifacts can use the general verifier's `verify_supervised` function.

The fixed-target audit additionally checks the compact transition table, paired sampling and initialization, exact-arm identity with the preceding study, paired outcomes and efficient-success denominators. Its `--skip-forward-inference` mode is included in general artifact verification and CI: it retains saved-model hashes and all dataset/metric checks while omitting bit-exact forward reproduction. Use `--study <directory> --allow-smoke` to audit a completed smoke structurally; this never makes it eligible for research gates. Incomplete or inconsistent fixed-target artifacts require inspection of their saved stop reason and partial evidence and are not accepted as complete studies by this audit.

The coverage audit also reconstructs the recorded exploratory collection, checks deduplicated current-state support, sampling/exposure membership, successor-query access and coverage denominators. Collection reconstruction is a read-only check of a fixed collector history; it does not train a network or add learner-policy evaluation episodes. It provides the same explicit smoke and skip-forward modes. The exhaustive arm's archived DDQN identity check distinguishes a changed coverage condition from an accidentally changed control implementation.

The equal-size audit checks the archived collected support and declared uniform-subset draw, equal local sampling schedules, each bank's direct exposure and successor coverage, and exact historical collected-arm identity. Its supported/outside-state diagnostics use each condition's own mask; full-bank fit gates remain separate. Neither the experiment nor its audit performs new collection.

The frozen-panel audit reconstructs prospective panel admission and exclusions,
checks archived model identity before/after evaluation, and reconciles raw
episode, paired and panel summaries. Its forward check uses only preselected
saved replay observations; it does not repeat the full policy evaluation.
`--skip-forward-inference` preserves the portable artifact/arithmetic checks.
No new training-fit or competence gate belongs to this evaluation-only study.

The bank-replication audit reconstructs three separately randomized collector
histories and their matched-size uniform draws, verifies within-pair local
sampling and each bank's direct/successor exposure, and reconciles final
per-bank outcomes before equal-bank pooling. It checks saved replay observations
without rerunning training or the complete final-policy evaluation. The three
banks share training layouts; learner seeds are not extra bank replications.

The map-replay audit checks reuse of those exact collected supports and archived
baseline policies, reconstructs independent map/within-map sampling streams,
and verifies exposure summaries and final paired behavior. It distinguishes
new treatment updates from historical baseline training and performs no new
collection, retraining or complete policy evaluation.

## Build a package

```bash
python -m build
```

CI builds both an sdist and a wheel, installs the wheel into an isolated target, and checks imports away from the source directory. Package discovery is explicit: it includes `q6`, `no_way_home`, and selected legacy support modules, with the root `config` module retained for compatibility. It excludes experiment outputs and historical training scripts. The static dashboard and legacy trainer commands should be used from the repository checkout.

## Experiment template

Copy this outline into `docs/experiments/<study>.md`. Mark it as a protocol or an exploratory note; use a separate results document when outcomes are available.

```text
Question and scope:
Primary hypothesis and outcome:
Baseline(s) and the one intended change:
Observations and information available to each policy:
Calibration/training seeds and budget:
Untouched evaluation seeds/scenarios and budget:
Interaction, compute, and wall-clock limits:
Paired unit of analysis and uncertainty calculation:
Success criterion, failure criterion, and stopping rule:
Exact command, configuration, source revision, and environment:
Expected artifacts and their location:
Known limitations and possible confounds:
Post-run deviations from this protocol, with reasons:
```

Keep calibration, training, and evaluation roles separate. If a seed or result has informed tuning, label it as observed and choose fresh evaluation data. Compare methods under stated interaction and compute budgets. For paired simulations, preserve pairing in the analysis and isolate policy randomness from world randomness. Report per-seed outcomes and distinguish training curves from fixed-policy evaluation.

## Reproduction record and artifacts

For a result intended for others to use, preserve:

- The complete invocation, resolved configuration, commit, dirty-state indicator, seeds, and checkpoint ancestry. A resume run should state what state was restored.
- Python, package, OS, device, and thread details; include `python -m pip freeze` output and any non-default deterministic settings.
- Raw per-seed metrics, evaluation trajectories, training curves where relevant, and the script that produces tables and dashboard summaries.
- A manifest of artifact paths, byte sizes, and SHA-256 hashes. Put large checkpoints in a release artifact or other persistent archive and document retrieval; do not commit credentials or machine-specific absolute paths.
- A clear status: smoke, exploratory pilot, or a completed comparison under the stated protocol. Include negative results, deviations, and uncertainty.

Use a new output directory for a new experiment. The legacy `training_runs/` directory is intentionally ignored, so its existence on the author's machine is not a public artifact archive. Small curated results under `experiments/` can be committed deliberately; inspect them before adding them. Keep old reports intact and append dated errata when their interpretation changes.

For a pull request, lead with the problem and resulting behavior, then list the relevant checks and remaining limitations. Cite the original methods you build on. Describe assistance and attribution where they matter, and avoid claiming algorithmic novelty from an implementation or renamed component alone.
