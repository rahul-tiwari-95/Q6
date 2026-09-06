# Competence milestone validation — 2026-09-06

This record covers the new A-only runner, exact reference, saved study and
dashboard extension. It complements the [earlier preview validation](2026-09-06.md).
Software correctness checks do not establish a research claim or independent
replication; the [result report](../experiments/competence_results_v1.md) states
the outcome and its limits.

## Runtime and software checks

- Main training and local tests: Python 3.12.13, Torch 2.8.0, NumPy 2.0.2,
  macOS arm64, one Torch CPU thread. Full resolved package versions are captured
  in the study's `environment.txt`.
- The documented seeded fast lane passed **299 tests, 3 deselected in 42.17s**.
  The three unchanged legacy training-heavy tests passed in the earlier
  milestone and were not repeated for this runner/dashboard addition.
- Six exact-reference tests include Bellman checks against actual environment
  transitions, blocked movement, action remapping, finite-horizon termination,
  shaping, and last-chance reachability.
- Six competence tests include exact continuation of learning across an
  evaluation checkpoint, evaluation RNG isolation, task sampling and canonical
  identity, diagnostic denominators, compressed logs, and incomplete-run gates.
- A genuine CLI smoke completed 192 training transitions in 2.91 seconds.
  It is marked ineligible for scientific gates and is not shipped as dashboard
  evidence. Short budgets and partial checkpoints cannot pass the declared gates.

## Main artifact checks

`python scripts/verify_pilot_artifacts.py` verified:

- All **62 main-study artifact hashes**, captured source/protocol hashes,
  **60 evaluation aggregate groups**, and **1,080,000 logged training transitions**.
- All nine files in the read-only prior-v2 diagnostic manifest.
- Both historical adaptation pilots' source/protocol hashes and 28 aggregate
  groups, plus all 26 provenance manifest files.

The main run has 45 complete scheduled checkpoints, 13,095 learner evaluation
episodes, 193,844 training episode records including partial final episodes,
and 192 saved trajectories. Its 2,169 reference CSV rows present **848 unique
reference samples**, reused across panels/conditions. Repeated presentation is
not new evidence. The dashboard JSON byte-matches the saved main result.

The source revision captured before training is clean `88fb9d2`. The protocol
was committed before execution. Later changes update the viewer, verification
harness and documentation; they do not change captured training sources or
historical artifacts. The late gate-footnote color adjustment makes unmet
results gray instead of green.

## Dashboard checks

`node scripts/check_dashboard.mjs` runs from any directory and passed on the
real exported study:

- All 192 saved recordings are reachable through selection handlers.
- All 60 aggregate rows and 36 condition/action-mode/controller/panel
  combinations render. Pre-action Q values are labeled separately from the
  grid's post-action state, including initial and terminal frames.
- Play/pause, automatic stop, reset, scrub, stop on track switch, refresh, and
  historical adaptation selection pass. Existing provenance data still load.
- The missing-study path was checked during development and displays an empty
  state rather than invented curves or zero results.

This harness uses DOM/canvas stubs, so it checks code paths rather than browser
layout. Native Chrome loaded the actual main dataset and showed the correct
three-condition scores, gate text, source links and action-value table. Selecting
the sixteen-task condition updated the replay and reference values. The initial
desktop screenshot showed the new hierarchy and cards without overflow. Native
scroll/capture automation became unreliable (`noWindowsAvailable`), so a complete
pixel review of the lower replay panel and responsive layouts was not established
in this milestone. The broader interaction checks above are harness results,
not claims of exhaustive native-browser coverage.

## Continuous integration

The workflow runs the fast lane on Python 3.10 and 3.12, saved-artifact integrity,
dashboard syntax and the same dashboard control harness. The Python 3.12 job
also builds and installs a wheel away from source. The full lane remains
scheduled/manual. Consult [PR #1](https://github.com/rahul-tiwari-95/Q6/pull/1/checks)
for the result on the precise submitted revision; configuration alone is not
evidence of a passing run.
