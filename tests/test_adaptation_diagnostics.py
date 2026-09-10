"""A planner must use visible rules and respect the same task boundaries."""

from dataclasses import replace

import numpy as np

from q6.diagnostics import RandomPolicy, ShortestPathPolicy
from q6.world import CollectionWorld, WorldConfig


def test_shortest_path_respects_walls_and_observed_mapping():
    for mapping in ((0, 1, 2, 3), (3, 2, 0, 1)):
        for seed in range(40):
            env = CollectionWorld(WorldConfig(action_mapping=mapping))
            observation, _ = env.reset(seed=seed)
            initial_distance = -env._potential() * env.config.size
            planner = ShortestPathPolicy(env.config.size)
            steps = 0
            while True:
                action = planner.act(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                assert not env.walls[env.position]
                steps += 1
                if terminated or truncated:
                    break
            assert terminated and not truncated
            assert steps == round(initial_distance)


def test_planner_cannot_skip_steps_or_extend_horizon():
    env = CollectionWorld(WorldConfig(wall_count=0, horizon=1))
    env.reset(seed=0)
    state = env.state_dict()
    state["position"] = [0, 0]
    state["pellets"] = np.zeros((5, 5), dtype=int).tolist()
    state["pellets"][4][4] = 1
    env = CollectionWorld.from_state(state)
    planner = ShortestPathPolicy(5)
    _, _, terminated, truncated, _ = env.step(planner.act(env.observe()))
    assert truncated and not terminated and env.elapsed == 1


def test_random_reference_is_seeded_independently():
    a, b = RandomPolicy(3, 910000), RandomPolicy(3, 910000)
    noise = RandomPolicy(9, 910001)
    expected, observed = [], []
    for _ in range(20):
        noise.act(np.zeros(1))
        expected.append(a.act(np.zeros(1)))
        observed.append(b.act(np.ones(1)))
    assert expected == observed
    assert len(set(expected)) > 1
