"""Bounded, read-only RL audit reproductions; first arg optionally changes repo root."""
import copy
import json
import sys
from collections import deque
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
import numpy as np
from environment.selfplay_env import SelfPlayGridworld
from utils.cher import CounterfactualHER

def setup(k=(1, 1), h=(8, 8), pellets=((1, 3), (0, 8), (8, 0), (8, 9))):
    env = SelfPlayGridworld(10, 0)
    env.reset(seed=0)
    env.grid[:] = env.EMPTY
    env.krishna_pos, env.hunter_pos = k, h
    env.pellet_positions = set(pellets)
    for p in pellets:
        env.grid[p] = env.PELLET
    env.grid[k], env.grid[h] = env.KRISHNA, env.HUNTER
    return env

results = {}
env = setup(h=(1, 2))
env.step({'krishna': 0, 'hunter': 3})
env.step({'krishna': 0, 'hunter': 3})
results['pellet_erasure'] = {'pellet_exists': (1, 3) in env.pellet_positions,
                           'observed_cell': int(env.grid[1, 3]), 'pellet_cell_id': env.PELLET}
env = setup()
original = copy.deepcopy(env)
state, info = env._get_state(), env._get_info()
ns, rew, done, trunc, ni = env.step({'krishna': 0, 'hunter': 0})
cf = CounterfactualHER().relabel([dict(state=state, context=np.zeros(3), action=0,
    reward=rew['krishna'], next_state=ns, next_context=np.zeros(3),
    done=done or trunc, info=info)])[0]
actual, real_reward, *_ = original.step({'krishna': cf[2], 'hunter': 0})
results['cher_transition'] = {'replacement_action': cf[2],
    'next_state_matches_actual_dynamics': bool(np.array_equal(cf[4], actual)),
    'synthetic_reward': cf[3], 'true_reward': real_reward['krishna']}
env = setup()
env.grid[1, 2] = env.WALL
cf = CounterfactualHER().relabel([dict(state=env._get_state(), context=np.zeros(3),
    action=0, reward=0, next_state=env._get_state(), next_context=np.zeros(3),
    done=False, info=env._get_info())])[0]
results['cher_wall_action'] = {'action': cf[2],
    'blocked': env._apply_action(env.krishna_pos, cf[2]) == env.krishna_pos}
env = setup(k=(1, 1), h=(1, 3))
other = copy.deepcopy(env)
other.invuln_timer = 10
same = bool(np.array_equal(env._get_state(), other._get_state()))
env.step({'krishna': 3, 'hunter': 2})
other.step({'krishna': 3, 'hunter': 2})
results['hidden_invulnerability'] = {'same_observation': same,
    'lives_after_same_action': [env.krishna_lives, other.krishna_lives]}
bad = []
env = SelfPlayGridworld()
for seed in range(2000):
    env.reset(seed=seed)
    visited = {env.krishna_pos}
    queue = deque(visited)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = x + dx, y + dy
            if 0 <= p[0] < 25 and 0 <= p[1] < 25 and p not in visited and env.grid[p] != env.WALL:
                visited.add(p)
                queue.append(p)
    if not env.pellet_positions <= visited:
        bad.append(seed)
results['map_connectivity'] = {'seeds_checked': 2000, 'unreachable_maps': len(bad),
    'example_seeds': bad[:10]}
print(json.dumps(results, indent=2))
