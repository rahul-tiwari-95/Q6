# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** with exact supervised action-value targets, the **unchanged 20,420-parameter network** reaches **85.42% greedy success on fresh layouts** (82.81–87.50% across three seeds). Every seed passes the declared supervised fresh-layout gate. The stricter training-fit diagnostic remains unmet, and routes still waste steps. This shows useful capacity under privileged targets and broad state coverage; it does not establish online-RL competence or isolate why earlier RL failed. Read the [result](docs/experiments/supervised_results_v1.md) and inspect its decisions in the dashboard. The next comparison holds data fixed to separate exact-target fitting from bootstrapped learning before adding memory.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.supervised --output /tmp/q6-supervised-smoke \
  --protocol-file docs/experiments/supervised_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. A smoke run checks execution and artifacts; its small training budget does not establish adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | Can this network solve fresh layouts with exact targets before relying on online RL? | [Current protocol](docs/experiments/supervised_protocol_v1.md), [results](docs/experiments/supervised_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current exact-target comparison in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
python -m q6.supervised --output experiments/supervised/my-reproduction \
  --protocol-file docs/experiments/supervised_protocol_v1.md
```

The [captured environment](experiments/supervised/pilot_v1/environment.txt) records the remaining versions. The runner uses the shipped historical RL checkpoints as frozen references. To display a reproduction locally, add `--dashboard dashboard/data/supervised.json`; this replaces the viewer's current data, so preserve the original artifact.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [exact supervision v1](experiments/supervised/pilot_v1/results.json), [A-only competence v1](docs/experiments/competence_results_v1.md), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The A-only RL diagnosis learned its selected fixed tasks but failed fresh-world competence. Exact supervision now demonstrates useful fresh-layout behavior with different data and target access. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html#supervised)** and select **Exact targets**. Compare training/fresh success, exhaustive action agreement and value error, then inspect a replay with learned and exact action values. Choose **Fresh layouts**, the final **30,000** updates, and greedy actions; switch **Supervised network** to **Historical frozen RL**. On the first fresh layout, selected before outcomes, all three supervised seeds finish in two steps while all three historical policies time out at 32. The complete panel, not that illustration alone, supplies the reported scores. The historical policies score 21.35% on this panel but are not matched for data, targets or computation.

**Learn one world** retains the preceding fixed-task/stream diagnosis; **Adaptation** and **Evidence** retain the earlier tracks. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves older run summaries. To index legacy training runs available on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Supervised validation](docs/validation/supervised-v1.md), [competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): tests, CI, artifact audits and browser coverage.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
