# Experience-coverage milestone validation — 2026-09-06

This record separates software and artifact checks from the scientific
interpretation in the [result report](../experiments/coverage_results_v1.md).
Both conditions retain privileged four-action transitions and detached
successor queries from the full training bank. This is an offline coverage
comparison, not an online-RL baseline.

## Protocol and software checks

The [protocol](../experiments/coverage_protocol_v1.md) was committed and pushed
at `552a7f7` before main execution, following a separate agent review with no
blocking methodological finding. It was locally declared, not externally
registered. The implementation reuses the previous DDQN update unchanged;
the exhaustive arm must reproduce the previous main DDQN weights and sampling
at every final seed.

Nine focused tests passed in **1.83s**. They check actual collector actions and
transitions, isolated random-number streams, deduplication, local/global batch
identities, unchanged exhaustive DDQN updates, detached outside-support
successor access, coverage denominators, both previous fresh-panel exclusions,
archived smoke integrity, preserved interrupted collection and explicit failure
when unique support is smaller than one batch.

The documented seeded fast lane passed **325 tests, 3 deselected in 43.82s**.
The three unchanged legacy training-heavy cases were not repeated for this
addition; their earlier validation remains archived.

The separate CLI smoke completed **32 collection episodes, 606 actual steps,
396/1,280 unique current states and 48 optimizer updates** in **0.893s**, with
**274.8 MB peak process RSS**. Its alternate layouts and small learning budget
make it ineligible for all research gates.

## Dashboard scope

Before main execution, the DOM/canvas-stub harness passed all **20 smoke
recordings**, 16 rollout aggregates, 40 state aggregates, both conditions and
action modes, two layout cells, five clock-bucket rows, seven interpretation
branches, final outcome cards, paired differences, pre-action Q labels,
playback and refresh. All five previous tracks remain available. This checks
code paths and displayed numerical values, not browser pixels.

Native visual review was unavailable: Computer Use failed to start its native
pipe. No pixel or responsive-layout verification is claimed for this track.
