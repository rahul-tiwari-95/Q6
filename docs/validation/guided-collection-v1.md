# Guided-collection validation

The single main run completed September 9, 2026. It is a valid negative result
for the declared mixture: see the [report](../experiments/guided_collection_results_v1.md).

## Declaration and source

The independently reviewed [protocol](../experiments/guided_collection_protocol_v1.md)
was pushed at `a668b3b81f38dcb93f69f37225e69fdee6e96531` before implementation
freeze. Main execution used clean, pushed source
`2a1fa03df615a3254c8f0a5e9933386b8a1eb2a5`. No deviations, main rerun or
outcome-dependent collector/checkpoint selection occurred. This is local
declaration and an independent agent audit, not external peer review.

Runtime: Python 3.12.13, Torch 2.8.0, NumPy 2.0.2, deterministic CPU with one
Torch thread, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
nice priority 10. The process was observed using about one core alongside
other Mac workloads. The 1,200-second admission and sampled 4-GiB process-RSS
guards are not hard OS or total-machine caps.

## Software and smoke

The final seeded fast lane passes **440 tests**, three unchanged long legacy
training tests deselected, in **104.31 seconds**. Final focused verification
passes **14 tests in 0.61 seconds**. Actual Python 3.10 compilation passes for
all auditors and changed Python files; JavaScript syntax and whitespace checks
pass. CI also exercises Python 3.10/3.12, portable evidence audits, dashboard
behavior and a Python 3.12 wheel installed away from the checkout.

New tests cover owned random streams, lowest-label guided ties, shared
deterministic routes, absent oracle access, no collector inference during random
episodes, frozen-weight mutation, nonfinite predictions, real terminal flags
under interruption, deduplication without frequency weighting, corrupted or
missing random log records, unequal replay populations and ineligible partial
run preservation.

The first broader check found three failures in the earlier familiar-start
tests: a new shared accounting hook changed the byte identity of
`q6/recorded_actions.py`, which that frozen diagnostic deliberately verifies.
The original file was restored exactly and the new per-arm accounting moved
into `q6/guided_collection.py`. The historical source check was not weakened,
and no historical result changed. All **34 familiar/guided focused tests**
then passed in **2.20 seconds**, followed by the green full fast lane.

Final smoke-v2 completes in **5.783 seconds**, sampled peak RSS
**515,964,928 bytes**: one bank, 4,096 complete episodes / 67,981 interactions,
24 learner updates, 1,890 action targets / 1,813 successor queries, two final
models, 24 learner plus 12 reference episodes and 44 recordings. It uses the
full 256 collection maps but a tiny learning budget and evaluation population;
the archived control still has 30,000 updates. It is execution evidence only.

Full and portable smoke audits pass in **3.87 / 3.76 seconds**, checking 98
manifest files, 64 historical inputs, all collected outcomes and random-slot
identities, recorded paths, exposure and replay evidence. The full audit
regenerates the 2,151 decisions in the distinct guided routes. The dashboard
smoke harness passes 44 new plus 2,938 prior recordings.

## Main integrity audit

The main completes in **186.297789 seconds**, peak process RSS
**601,538,560 bytes (0.560 GiB)**, with **792,284 resource checks**. It performs
270,000 new updates, 12,288 complete collection episodes / 203,113 interactions,
27,648 learner plus 3,584 unique reference evaluations, and preserves 400
preselected recordings.

The [independent full auditor](../../scripts/audit_guided_collection.py) passes
in **21.16 seconds**. It verifies:

- All **161 manifest files / 84 historical input records**, including captured
  source, original random logs, copied controls and collector provenance.
- All **163,840 state rows / 655,360 transition entries**, independently
  reconstructed geometry and rewards, with 480 direct kernel-clone checks.
- Every mixed collection step, including **151,489** exactly reproduced random
  steps. All guided duplicates match their 256 canonical routes across banks;
  2,151 unique saved collector decisions regenerate exactly.
- Both arms' distinct recorded tables, missing-action sentinels and successor
  closure; independent route reachability for all **1,536 bank/arm/start rows**.
- All **54 snapshots**, matching learner initializations, unchanged frozen
  controls and collector, and **540,000** reconstructed historical/new sampler
  updates, including each arm's own support and exposure counts.
- **21,322,723** new action targets: 960,030 terminal plus 20,362,693 nonterminal
  queries, all inside recorded support. Historical counts remain separate.
- All **31,232 evaluation episodes**, paired arithmetic, 2,700 loss windows,
  4,608 per-map exposure rows and 90 clock-exposure rows.
- All **304 fresh recordings / 4,549 steps** and **96 collection recordings /
  1,688 steps**, with selected fresh prediction regeneration. The full learned
  policy evaluation and training are not repeated.

The integrated portable verifier passes every shipped study. Its new-study
audit takes **19.62 seconds**, retaining hashes, recorded predictions,
transitions, sampler reconstruction and result arithmetic while omitting
forward regeneration. All historical artifacts remain byte-identical.

## Dashboard and publication

All **3,338 recordings** pass the DOM/canvas harness: 400 new and 2,938 prior.
Checks cover collection/fresh playback separation, fixed map/slot selection,
rule visibility, costs and historical collector inputs, action-target counts,
route ceilings, result reductions and missing/ineligible/incomplete states.
No UI changes were required after main execution.

The served export is **11,270,115 bytes**, SHA-256
`cb6ac9f0789815c6329d157db584b6280d2a6ca5a2ed61764bf6ce420a5de1ea`;
HTTP bytes at localhost:8080 exactly match the file. Served HTML, JavaScript
and CSS also match. Native CUA startup failed, so pixel and responsive visual
QA are not claimed.

Results, raw evidence, README and roadmap are published through the research
branch. Default `main` receives the README update only; PR #1 stays open and
the license remains undecided. Final publication CI links are recorded in
[PR #1](https://github.com/rahul-tiwari-95/Q6/pull/1).

```bash
python scripts/verify_pilot_artifacts.py
python scripts/audit_guided_collection.py --study experiments/guided_collection/pilot_v1
node scripts/check_dashboard.mjs
```
