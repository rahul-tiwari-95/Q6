# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** the uniform-support efficiency advantage repeats across **three independently collected, matched-size bank pairs**: **+17.12, +27.67 and +23.31 percentage points**. Across eight new 64-map panels, uniform support reaches **65.47% efficient success versus 42.77%** collected, with **83.57% versus 78.54% success** and **11.51 versus 15.90 mean steps**. Efficiency improves in every bank-panel mean, though one learner-panel cell reverses. Next: test map-balanced replay on the same collected states. Read the [bank-replication result](docs/experiments/bank_replication_results_v1.md). This remains privileged offline evidence, not online-RL competence.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.bank_replication --output /tmp/q6-bank-replication-smoke \
  --protocol-file docs/experiments/bank_replication_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. This smoke collects three tiny bank pairs, performs 144 training updates and evaluates four alternate maps. It checks execution and does not establish competence, adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | Does the efficiency advantage repeat across independently drawn, matched-size banks? | [Current protocol](docs/experiments/bank_replication_protocol_v1.md), [results](docs/experiments/bank_replication_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current bank-replication comparison in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.bank_replication \
  --output experiments/bank_replication/my-reproduction \
  --protocol-file docs/experiments/bank_replication_protocol_v1.md
```

The [captured environment](experiments/bank_replication/pilot_v1/environment.txt) records remaining versions. The runner freezes all six supports before training, completes all 18 fits before policy evaluation and preserves 90 snapshots. It runs sequentially on one CPU thread with a sampled 4 GiB process-RSS stop guard and a 1,200-second admission cap. The main comparison took **4.06 minutes** and peaked at **0.430 GiB**. To display a reproduction locally, add `--dashboard dashboard/data/bank_replication.json`; this replaces the viewer's current data, so preserve the original artifact. The [preceding panel protocol](docs/experiments/panel_evaluation_protocol_v1.md) reproduces the inference-only robustness study.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [bank replications v1](experiments/bank_replication/pilot_v1/results.json), [panel robustness v1](experiments/panel_evaluation/pilot_v1/results.json), [equal-size banks v1](experiments/equal_support/pilot_v1/results.json), [state coverage v1](experiments/coverage/pilot_v1/results.json), [paired fixed-data targets v1](experiments/fixed_targets/pilot_v1/results.json), [exact supervision v1](docs/experiments/supervised_results_v1.md), [A-only competence v1](docs/experiments/competence_results_v1.md), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The A-only RL diagnosis learned its selected fixed tasks but failed fresh-world competence. Subsequent studies establish useful offline behavior under broad coverage, first with exact targets and then bootstrapping. The coverage control found weaker transfer from collected states on its panel. At equal bank size, uniform support improves route efficiency while success is close; the repeated collected control exposes panel sensitivity. The prospective eight-panel evaluation found an efficiency advantage in every panel mean for the original policies. Three new bank pairs now replicate that advantage; the next intervention changes replay while holding collected support fixed. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html#coverage)** and select **Experience coverage → Bank replications**. Compare each bank's efficiency effect and the equal-bank mean, then inspect map/clock coverage, all eight evaluation panels and loss curves. The primary contrast stays greedy when replay mode changes. Historical threshold lines are descriptive reference levels. Switch banks, panels, controllers and learner seeds to inspect **304 preselected recordings**, with active rules and learned/exact pre-action values visible.

The study selector preserves **Panel robustness**, **Equal-size banks** and **Exhaustive vs collected**, each with its own evidence. **Fixed-data targets**, **Exact targets**, **Learn one world**, **Adaptation** and **Evidence** retain earlier tracks; all **794 earlier learner-study recordings** remain accessible. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves older summaries. To index legacy training runs on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Bank-replication validation](docs/validation/bank-replication-v1.md), [panel-evaluation validation](docs/validation/panel-evaluation-v1.md), [equal-size validation](docs/validation/equal-support-v1.md), [coverage validation](docs/validation/coverage-v1.md), [fixed-target validation](docs/validation/fixed-targets-v1.md), [supervised validation](docs/validation/supervised-v1.md), [competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): tests, CI, artifact audits and browser coverage.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
