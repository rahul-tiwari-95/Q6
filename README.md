# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** frozen collected, uniform-subset and exhaustive policies evaluated on **eight new 64-map panels** reach **78.19%, 80.73% and 84.90% greedy success**. Uniform support improves efficient success over equal-size collected support on **all eight panel means** (**61.72% versus 46.42%**, +15.30 points pooled), with shorter episodes throughout. Success reverses on one panel; these are the same fixed policies, not independent bank replications. No training or collection occurred. Next: replicate support banks before choosing a collection/replay intervention. Read the [panel robustness result](docs/experiments/panel_evaluation_results_v1.md). This remains privileged offline evidence, not online-RL competence.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.panel_evaluation --output /tmp/q6-panel-evaluation-smoke \
  --protocol-file docs/experiments/panel_evaluation_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. This smoke loads shipped frozen checkpoints and evaluates four alternate maps; it performs no training and does not establish adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | Does the fixed-policy efficiency advantage repeat across fresh panels? | [Current protocol](docs/experiments/panel_evaluation_protocol_v1.md), [results](docs/experiments/panel_evaluation_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current frozen-policy panel evaluation in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python -m q6.panel_evaluation \
  --output experiments/panel_evaluation/my-reproduction \
  --protocol-file docs/experiments/panel_evaluation_protocol_v1.md
```

The [captured environment](experiments/panel_evaluation/pilot_v1/environment.txt) records remaining versions. The runner loads nine shipped final policies, selects all panels before predictions and checks model hashes before/after every panel. It performs no training, support draws or collection. It runs sequentially on one CPU thread with a sampled 4 GiB process-RSS stop guard. The main evaluation took **31 seconds** and peaked at **0.226 GiB**. To display a reproduction locally, add `--dashboard dashboard/data/panel_evaluation.json`; this replaces the viewer's current data, so preserve the original artifact. Read the [equal-size protocol](docs/experiments/equal_support_protocol_v1.md) to reproduce the preceding training comparison.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [panel robustness v1](experiments/panel_evaluation/pilot_v1/results.json), [equal-size banks v1](experiments/equal_support/pilot_v1/results.json), [state coverage v1](experiments/coverage/pilot_v1/results.json), [paired fixed-data targets v1](experiments/fixed_targets/pilot_v1/results.json), [exact supervision v1](docs/experiments/supervised_results_v1.md), [A-only competence v1](docs/experiments/competence_results_v1.md), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The A-only RL diagnosis learned its selected fixed tasks but failed fresh-world competence. Subsequent studies establish useful offline behavior under broad coverage, first with exact targets and then bootstrapping. The coverage control found weaker transfer from collected states on its panel. At equal bank size, uniform support improves route efficiency while success is close; the repeated collected control exposes panel sensitivity. The prospective eight-panel evaluation now finds an efficiency advantage in every panel mean, conditional on the same support banks and policies. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html#coverage)** and select **Experience coverage → Panel robustness**. Compare success, efficient success and mean steps across all eight panels; inspect pooled values and panel ranges. The paired-efficiency chart shows uniform minus collected, with descriptive sign counts. Historical threshold lines are reference levels, not new gates. Switch panels, controllers and learner seeds to inspect 160 preselected recordings, with active rules and learned/exact pre-action values visible.

The study selector preserves **Equal-size banks** and **Exhaustive vs collected**, each with its own panel and diagnostics. **Fixed-data targets**, **Exact targets**, **Learn one world**, **Adaptation** and **Evidence** retain earlier tracks. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves older summaries. To index legacy training runs on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Panel-evaluation validation](docs/validation/panel-evaluation-v1.md), [equal-size validation](docs/validation/equal-support-v1.md), [coverage validation](docs/validation/coverage-v1.md), [fixed-target validation](docs/validation/fixed-targets-v1.md), [supervised validation](docs/validation/supervised-v1.md), [competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): tests, CI, artifact audits and browser coverage.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
