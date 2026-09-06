# Equal-size bank milestone validation — 2026-09-06

This record separates implementation and evidence checks from the scientific
interpretation in the [result report](../experiments/equal_support_results_v1.md).
The study retains privileged four-action transitions and detached full-bank
successor access. It performs no new collection and does not establish an
online-RL baseline.

## Protocol and implementation scope

The [protocol](../experiments/equal_support_protocol_v1.md) was independently
reviewed, committed and pushed at `03f0ff8` before main execution. It was locally
declared, not externally registered. A shared runner extension supplies bank
selection, provenance and diagnostics; DDQN update mathematics, the network
and the historical coverage entry point remain unchanged.

The main control requires exact reproduction of the preceding collected arm's
initial/final online weights, final target weights and global sampling digest
and count vector. Equal cardinality additionally requires identical local
sampling digests/count vectors across the new arms. Global states intentionally
differ. These checks establish execution consistency, not independent bank
replication.

## Dashboard scope

The Experience coverage tab now has a study selector. Equal-size banks is the
new default; the previous exhaustive-versus-collected comparison remains
available with its own fresh panel and interpretation. Both conditions' bank
composition is shown together, while a selector reuses the detailed map/time
views. Own-support diagnostics remain separate from full-bank fit gates.

Native browser review was unavailable: Computer Use failed to start its native
pipe. No pixel or responsive-layout verification is claimed. DOM/canvas-stub
checks and HTTP serving checks are separate from visual layout validation.

## Pre-execution checks

The documented seeded fast lane passed **331 tests, 3 deselected in 42.69s**.
The unchanged three legacy training-heavy cases were not repeated. All **six
new equal-size tests plus nine legacy coverage tests passed in 2.21s**. They
check the owned uniform RNG and exact draw, archive support/mask integrity,
three prior-panel exclusions, each arm's own diagnostic mask, equal local
sampling with distinct global rows, absence of collection and post-diagnostic
resource-cap disqualification.

The final separate CLI smoke used two alternate maps per panel and synthetic
every-third-row collected support. It completed **48 updates / 3,072 state
presentations**, with **427 states per arm, zero collection steps**, in
**0.822s**, using **284.49 MB peak process RSS**. It is ineligible for all gates.

The independent audit passed the final smoke in **0.436s** with regenerated
forward predictions and **0.318s** in portable mode. It verified 45 manifest
files, 10,240 exact values, 5,120 transitions, 960 isolated kernel steps,
support selection, local/global sampling, model/prediction hashes and raw
metrics/gates/paired/own-support diagnostics. Historical artifact verification
also passed after generalizing the shared raw-table audit helper.

The dashboard smoke harness checks all 20 new recordings, both supports,
equal-size/overlap counts, own/outside-support diagnostics, seven interpretation
branches, action modes, paired outcomes, study switching and playback stop.
All 510 earlier learned-study recordings and historical controls remain
available. These checks validate code paths and displayed numbers, not pixels.
