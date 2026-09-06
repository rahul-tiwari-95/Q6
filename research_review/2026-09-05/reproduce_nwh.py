"""Read-only Q6 NWH audit reproductions. Run from repository root:
python3 research_review/2026-09-05/reproduce_nwh.py
All generated artifacts go to /tmp, no source or original results are edited.
The threshold comparison is EXPLORATORY, not a confirmatory experiment.
"""
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path.cwd()))
import numpy as np
from no_way_home.world import WorldConfig, run
from no_way_home.policies import never_mitigate, lineage_aware_heuristic
from no_way_home.institutions import instrument_mixed
from no_way_home.learning import TabularQMandateLearner, feature_raw_rate, RATE_BINS
from no_way_home.metrics import need_shortfall_per_10k
from no_way_home.run_beta_sweep_v1 import was_fooled

betas = np.array([0., .25, .5, .75, 1.])
cfg = WorldConfig(n_ticks=799)
rows = []
for seed in range(20000, 20050):
    state = run(cfg, never_mitigate, seed)
    flags = np.zeros(5, dtype=bool)
    for tick in range(1, 800):
        recent = [m for m in state.messages.messages if tick-30 <= m.tick < tick]
        raw = len(recent) / 30
        unique = len({m.origin_id for m in recent}) / 30
        flags |= (1-betas)*raw + betas*unique > .30
    rows.append(flags.astype(int).tolist())
print('beta counts reconstructed entirely from never-mitigate messages:', np.sum(rows, axis=0).tolist())
print('nonmonotone paired rows:', sum(any(r[i] < r[i+1] for i in range(4)) for r in rows))
checks = 0
for j, seed in enumerate(range(20000, 20010)):
    for i, beta in enumerate(betas):
        checks += was_fooled(run(cfg, instrument_mixed(float(beta)), seed)) == bool(rows[j][i])
print('direct policy checks matching analytical reconstruction:', checks, '/ 50')
Path('/tmp/q6-nwh-beta-paired.json').write_text(json.dumps({
    'seeds': list(range(20000,20050)), 'betas': betas.tolist(), 'outcomes': rows}, indent=2))

# With binary outcomes and equal arm sizes, JT is linear in the arm totals.
# Reversed group ordering matches the implementation's decreasing alternative.
y = np.array(rows)
n, k = y.shape
weights = np.arange(k-1, -k, -2)
observed = (y.sum(axis=0)*weights).sum()
rng = np.random.default_rng(2026)
exceedances = 0
for _ in range(9999):
    perm = np.take_along_axis(y, np.argsort(rng.random(y.shape), axis=1), axis=1)
    exceedances += (perm.sum(axis=0)*weights).sum() >= observed
print('within-seed permutation p (+1 correction):', (exceedances+1)/10000)
print('equivalent observed JT decreasing:', .5*n*n*(k*(k-1)//2) + .5*n*observed)

cfg_ids = WorldConfig(n_ticks=20, report_sick_threshold=1)
a = run(cfg_ids, never_mitigate, 0)
b = run(cfg_ids, never_mitigate, 0)
print('same seed events identical:', a.events == b.events,
      'message log identical:', a.messages.messages == b.messages.messages)
print('first origin IDs:', a.messages.messages[0].origin_id, b.messages.messages[0].origin_id)
learner = TabularQMandateLearner(feature_fns=[feature_raw_rate], bin_edges=[RATE_BINS])
terminal_cfg = WorldConfig(n_ticks=1, starting_food_stock=0., base_food_yield_per_agent=0.,
    medicine_production_per_tick=0., starting_medicine_stock=0., base_sickness_prob=1.)
for seed in range(2):
    terminal = run(terminal_cfg, learner, seed)
print('terminal shortfall:', terminal.events[-1]['shortfall_this_tick'],
      'Q updates across 2 one-step runs:', sum(learner.visit_counts.values()))

print('EXPLORATORY controls, seeds 15000-15009, 2000 ticks; not confirmatory')
cfg = WorldConfig()
controls = []
for name, policy in [
    ('aware .30', lineage_aware_heuristic),
    ('raw .60', instrument_mixed(0., .6)),
    ('raw .80', instrument_mixed(0., .8)),
    ('raw 1.00', instrument_mixed(0., 1.)),
    ('never', never_mitigate),
]:
    states = [run(cfg, policy, seed) for seed in range(15000,15010)]
    fooled = sum(map(was_fooled, states))
    shortfalls = [need_shortfall_per_10k(s) for s in states]
    print(name, 'fooled', fooled, '/10', 'shortfall', round(mean(shortfalls)))
    controls.append({'name': name, 'fooled': fooled, 'shortfalls': shortfalls})
Path('/tmp/q6-nwh-exploratory-controls.json').write_text(json.dumps({
    'status': 'EXPLORATORY', 'seeds': list(range(15000,15010)), 'controls': controls}, indent=2))
