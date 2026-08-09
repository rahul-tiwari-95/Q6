# No Way Home

A deterministic multi-agent gridworld testing whether a system built from simple, checkable
parts (noisy voting, raw message counting, tabular Q-learning) can make good collective
decisions under a hidden regime shift — and precisely diagnosing what breaks when it doesn't.
Second parallel research thread in Q6, alongside the original self-play work on `main`/`v7`/`v8`.

Branch: `no-way-home`. This file is the folder's front door — start here, then follow the links
below to whichever depth you actually need.

## Start here

| If you want... | Read |
|---|---|
| A plain-language walkthrough with examples, no formal notation | [`README_FOR_DUMMIES.md`](README_FOR_DUMMIES.md) |
| What's currently being worked on, right now | [`ENGINEERING_NWH_PHASE_PLAN.md`](ENGINEERING_NWH_PHASE_PLAN.md) |
| The formal spec and invariants (I-1 through I-12) | [`PREREGISTRATION.md`](PREREGISTRATION.md) |
| The environment redesign proposal (channels, manipulability, the β-sweep experiment) | [`ENVIRONMENT_REDESIGN.md`](ENVIRONMENT_REDESIGN.md) |
| The full experiment history, what broke and what fixed it, increment by increment | [`results/README.md`](results/README.md) |
| Sources used and why each one mattered | [`../CITATIONS.md`](../CITATIONS.md) |

## Current status (2026-08-09)

Five increments of the base simulation are built and tested (physical economy → message/lineage
layer → ballot/election/institution → learning citizen), each with real bugs found and fixed
along the way, documented honestly rather than smoothed over. A research-backed redesign
(`ENVIRONMENT_REDESIGN.md`) was approved and its migration path is underway: Increments 1-2 are
complete (schema tags, the β-mixing instrument, both required control arms), and Increment 3 —
the actual pre-registered dose-response experiment — is in its pre-registration stage. See
`ENGINEERING_NWH_PHASE_PLAN.md` for the live TODO list (capped at 3 items) and exact current
state; this README won't be kept in lockstep with it on every commit, that file is the source of
truth for "what's happening right now."

## Running things

```bash
python3 -m pytest tests/test_no_way_home_smoke.py tests/test_stats.py -q   # the suite
python3 -m no_way_home.run_smoke_test          # base policy comparison
python3 -m no_way_home.run_election_test       # ballot/election/institution test
python3 -m no_way_home.run_learning_citizen    # the Q-learning citizen
python3 -m no_way_home.run_pilot_beta_sweep    # non-confirmatory pilot for Increment 3
python3 -m no_way_home.run_power_calculation   # power/precision calc feeding Increment 3's N
```

Each writes its own report to `results/`. `results/README.md` is the hand-maintained narrative
index — read that first if you want the story, not just the numbers.

## Layout

`world.py` (kernel) → `messages.py` (report/forward lineage) → `channels.py` (manipulability
tags) → `policies.py` (scripted decision rules) → `institutions.py` (ballot/election/executor)
→ `learning.py` (the Q-learning citizen) → `stats.py` (Jonckheere-Terpstra trend test) →
`metrics.py` (pure functions over the event log). `tests/` (one level up, in the repo root)
holds `test_no_way_home_smoke.py` and `test_stats.py`.
