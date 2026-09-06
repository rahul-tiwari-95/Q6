# Q6

**Small, inspectable experiments in learning across changing worlds.**

Q6 studies how a neural agent adapts when a task changes, and what it retains when an earlier task returns. The main experiment follows **A → B → A** in a compact collection world. A separate No Way Home experiment compares ways of counting repeated evidence in a synthetic resource-allocation task.

This is a public research preview; a license has not yet been selected. Its earlier Krishna–Hunter self-play work, failed experiments, and methodological corrections remain available as a research history.

**Latest finding:** the [A-only competence study](docs/experiments/competence_results_v1.md) completed **1,080,000 training transitions**. All three learner seeds reached **100% greedy success on both selected fixed training sets** (one task and sixteen tasks). Fresh-world success remained **9.90%, 22.40%, and 19.79%** for one-task, sixteen-task, and stream training; none passed the 70%-in-every-seed gate. Q6 can learn these familiar tasks, but useful performance across new worlds is still missing. The next diagnosis separates learning reliable navigation targets from the online RL procedure before adding memory.

## Start on CPU

Use Python 3.10 or later; Python 3.12 is a practical starting point. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Short execution check; writes to a new temporary output directory.
python -m q6.competence --output /tmp/q6-competence-smoke \
  --protocol-file docs/experiments/competence_protocol_v1.md --smoke
```

Choose a fresh `--output` directory when repeating a run. A smoke run checks execution and artifacts; its small training budget does not establish adaptation or retention.

## Experiments

| Track | Question | Start here |
| --- | --- | --- |
| **Competence → adaptation — primary** | Can the learner solve new A tasks reliably enough to study B and recovery of A? | [Current protocol](docs/experiments/competence_protocol_v1.md), [results](docs/experiments/competence_results_v1.md), [`q6/`](q6/) |
| **Provenance — bounded companion** | Does deduplicating message origins improve decisions compared with independently calibrated raw and decayed counts? | [Protocol](docs/experiments/provenance-protocol.md), [errata](docs/experiments/provenance-errata.md), [`no_way_home/`](no_way_home/) |

Reproduce the current A-only comparison in a fresh directory:

```bash
# Match the main run's core dependencies in Python 3.12.
python -m pip install 'torch==2.8.0' 'numpy==2.0.2'
python -m q6.competence --output experiments/competence/my-reproduction \
  --protocol-file docs/experiments/competence_protocol_v1.md
```

The [captured environment](experiments/competence/pilot_v1/environment.txt) records the remaining versions. To display a reproduction locally, add `--dashboard dashboard/data/competence.json`; this replaces the viewer's current data, so preserve the original artifact.

Run the fixed-budget provenance pilot:

```bash
python -m no_way_home.run_provenance_release \
  --out experiments/provenance/my-reproduction
```

Stored studies: [competence v1](experiments/competence/pilot_v1/results.json), [adaptation v2](docs/experiments/adaptation_pilot_v2.md), [archived adaptation v1](docs/experiments/adaptation_pilot_v1.md), and [provenance](experiments/provenance/release-pilot-v1/RESULTS.md). Both adaptation pilots failed initial competence; their later A→B→A outcomes do not establish forgetting or recovery. The separate A-only diagnosis learned its fixed support but failed fresh-world competence. Each milestone stops at its declared budget. Runners refuse to overwrite an existing study directory; read the protocol before changing its seeds or budget.

## Inspect results

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **[the current lab](http://127.0.0.1:8080/dashboard/lab.html)** and select **Learn one world** for familiar-versus-fresh learning curves. Choose **Sixteen repeated tasks**, switch the replay between **Training tasks** and **Fresh maps**, and compare the learned policy with **Shortest path**. Step through the recorded decisions to see learned and exact action values. The [historical registry](http://127.0.0.1:8080/dashboard/index.html) preserves earlier run summaries. To index legacy training runs available on your machine, run `python dashboard/scan.py` before starting the server.

Historical summaries are not a complete artifact archive: many referenced CSV files, checkpoints, and replays are absent from a fresh checkout. Claims from the earlier reports should be read with the [September 2026 review](research_review/2026-09-05/README.md), which documents confounds, bugs, and unsupported interpretations. The new pilots also need independent seeds and fair baselines before supporting general conclusions. No Way Home is a synthetic decision model, not empirical evidence about human institutions.

## Develop and contribute

- [Contribution guide](CONTRIBUTING.md): fast and full test commands, experiment template, and reproducibility requirements.
- [Documentation index](docs/README.md) and [roadmap](docs/roadmap.md): current scope and criteria for the next experiment.
- [Competence validation](docs/validation/competence-v1.md) and [earlier preview validation](docs/validation/2026-09-06.md): test coverage, CI, artifact integrity, and runtime.
- [Training supervisor](orchestrator/README.md): local job logging, crash restart, and resume conventions.
- [Citation metadata](CITATION.cff): cite the repository and the exact revision/artifact used.

The package installs the current experiment modules and selected legacy support modules. Historical trainer scripts and the dashboard are documented as commands run from this checkout; their APIs are not a stable library contract.

## Research history

The [version index](versions/README.md), [articles](articles/), [original Q6 narrative](Q6.md), [No Way Home design](Q6%20No%20Way%20Home.md), and [archive](archive/README.md) preserve the project's progression. Older ablations and the PPO port live on the [`v7-ablations`](https://github.com/rahul-tiwari-95/Q6/tree/v7-ablations) and [`v8`](https://github.com/rahul-tiwari-95/Q6/tree/v8) branches. These are historical experiments, not interchangeable controlled comparisons. The [`longer_memory/`](longer_memory/README.md) folder contains old planning notes, not an implemented memory architecture.
