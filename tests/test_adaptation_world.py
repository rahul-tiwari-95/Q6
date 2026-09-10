"""Transition and state invariants for the small adaptation kernel."""

import json
from dataclasses import replace

import numpy as np
import pytest

from q6.world import CollectionWorld, WorldConfig, reachable


def configured(position=(2, 2), pellet=(0, 0), *, horizon=8, mapping=(0, 1, 2, 3)):
    env = CollectionWorld(WorldConfig(wall_count=0, horizon=horizon, action_mapping=mapping))
    env.reset(seed=0)
    state = env.state_dict()
    state["position"] = list(position)
    state["pellets"] = np.zeros((5, 5), dtype=int).tolist()
    state["pellets"][pellet[0]][pellet[1]] = 1
    return CollectionWorld.from_state(state)


def test_all_generated_objectives_reachable_and_layers_disjoint():
    for seed in range(200):
        env = CollectionWorld()
        env.reset(seed=seed)
        component = reachable(env.walls, env.position)
        assert all(tuple(p) in component for p in np.argwhere(env.pellets))
        assert not np.any(env.walls & env.pellets)
        assert not env.walls[env.position] and not env.pellets[env.position]
        assert env.walls.sum() == 4 and env.pellets.sum() == 1


def test_world_rule_changes_only_action_interpretation():
    a = configured()
    b = configured(mapping=(3, 2, 0, 1))
    assert np.array_equal(a.walls, b.walls) and np.array_equal(a.pellets, b.pellets)
    assert not np.array_equal(a.observe(), b.observe())  # the rule is visible
    a.step(0)
    b.step(0)
    assert a.position == (1, 2) and b.position == (2, 3)


def test_walls_block_motion_without_erasing_other_layers():
    env = configured()
    env.walls[1, 2] = True
    pellets, walls = env.pellets.copy(), env.walls.copy()
    env.step(0)
    assert env.position == (2, 2)
    np.testing.assert_array_equal(env.pellets, pellets)
    np.testing.assert_array_equal(env.walls, walls)


def test_finite_horizon_and_last_step_success_are_unambiguous():
    timeout = configured(horizon=1)
    before = timeout.observe()
    state, _, term, trunc, _ = timeout.step(1)
    assert not term and trunc
    assert not np.array_equal(before, state)
    assert state[3 * 25] == 0
    with pytest.raises(RuntimeError):
        timeout.step(1)
    success = configured(position=(0, 1), horizon=1)
    _, _, term, trunc, info = success.step(2)
    assert term and not trunc and info["pellets_remaining"] == 0
    assert info["base_reward"] == pytest.approx(0.99)


def test_state_roundtrip_clones_rng_and_has_no_shared_layers():
    env = CollectionWorld(seed=12)
    env.reset()
    clone = CollectionWorld.from_state(json.loads(json.dumps(env.state_dict())))
    for action in (0, 3, 1):
        if env.terminated or env.truncated:
            break
        left, right = env.step(action), clone.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    np.testing.assert_array_equal(env.reset()[0], clone.reset()[0])
    clone.walls[:] = False
    assert env.walls.sum() == 4


def test_potential_shaping_preserves_discounted_episode_return_offset():
    # Two very different trajectories have the same shaping-only offset from
    # the same start, including a horizon timeout and a successful route.
    for actions in ([0, 0, 2, 2], [1] * 8):
        env = configured()
        phi_start = env._potential()
        offset = 0.0
        for index, action in enumerate(actions):
            _, reward, term, trunc, info = env.step(action)
            offset += env.config.gamma**index * (reward - info["base_reward"])
            if term or trunc:
                break
        assert term or trunc
        assert offset == pytest.approx(-env.config.shaping_weight * phi_start)


def test_seeded_reset_is_independent_of_other_worlds():
    a, b, noise = CollectionWorld(seed=4), CollectionWorld(seed=4), CollectionWorld(seed=9)
    for _ in range(5):
        noise.reset()
        np.testing.assert_array_equal(a.reset()[0], b.reset()[0])


@pytest.mark.parametrize("action", [-1, 4, 1.5, True])
def test_bad_actions_rejected(action):
    env = configured()
    with pytest.raises(ValueError):
        env.step(action)
