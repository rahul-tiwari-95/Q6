"""Independent finite-horizon DP versus actual kernel transitions and returns."""
from dataclasses import replace

import numpy as np
import pytest

from q6.optimal import VisibleOptimalQ
from q6.world import CollectionWorld, WorldConfig


def configured(position=(2, 2), goal=(0, 0), horizon=8):
    env = CollectionWorld(WorldConfig(wall_count=0, horizon=horizon))
    env.reset(seed=1)
    state = env.state_dict()
    state['position'] = list(position)
    state['pellets'] = np.zeros((5, 5), dtype=int).tolist()
    state['pellets'][goal[0]][goal[1]] = 1
    return CollectionWorld.from_state(state)


def test_independent_dp_satisfies_actual_transition_bellman_equation():
    """Cross-check remapping, wall/boundary blocks, success and timeout."""
    saw = {'blocked': False, 'success': False, 'timeout': False}
    for mapping in ((0, 1, 2, 3), (3, 2, 0, 1)):
        cfg = WorldConfig(action_mapping=mapping)
        reference = VisibleOptimalQ(cfg)
        for map_seed in range(3):
            env = CollectionWorld(cfg)
            env.reset(seed=map_seed)
            for position in np.argwhere(~env.walls & ~env.pellets):
                for remaining in (1, 2, cfg.horizon):
                    state = env.clone()
                    state.position = tuple(position)
                    state.elapsed = cfg.horizon - remaining
                    expected = reference.q_values(state.observe())
                    for action in range(4):
                        nxt = state.clone()
                        observation, reward, terminated, truncated, _ = nxt.step(action)
                        target = reward if terminated or truncated else reward + cfg.gamma * reference.q_values(observation).max()
                        assert expected[action] == pytest.approx(target, abs=1e-10)
                        saw['blocked'] |= nxt.position == state.position
                        saw['success'] |= terminated
                        saw['timeout'] |= truncated
    assert all(saw.values())


def test_known_route_and_unwinnable_timeout_have_correct_discounted_values():
    env = configured(position=(0, 2), goal=(0, 0), horizon=8)
    reference = VisibleOptimalQ(env.config)
    q = reference.q_values(env.observe())
    expected_base = -0.01 + env.config.gamma * 0.99
    assert q[2] == pytest.approx(expected_base + 0.2 * 2 / 5)
    assert reference.act(env.observe()) == 2
    assert reference.can_finish(env.observe())
    short = configured(position=(0, 2), goal=(0, 0), horizon=1)
    short_reference = VisibleOptimalQ(short.config)
    np.testing.assert_allclose(short_reference.q_values(short.observe()), -0.01 + 0.2 * 2 / 5)
    assert not short_reference.can_finish(short.observe())
    observation, _, _, truncated, _ = short.step(0)
    assert truncated
    np.testing.assert_array_equal(short_reference.q_values(observation), np.zeros(4))
    with pytest.raises(ValueError, match='after success or timeout'):
        short_reference.act(observation)


def test_shaping_offset_and_optimal_discounted_rollout_match_exact_q():
    env = configured(position=(2, 2), goal=(0, 0))
    reference = VisibleOptimalQ(env.config)
    unshaped = VisibleOptimalQ(replace(env.config, shaping_weight=0))
    obs = env.observe()
    initial = reference.q_values(obs).max()
    offset = -env.config.shaping_weight * env._potential()
    np.testing.assert_allclose(reference.q_values(obs) - unshaped.q_values(obs), offset)
    discounted = 0
    for tick in range(env.config.horizon):
        obs, reward, terminated, truncated, _ = env.step(reference.act(obs))
        discounted += env.config.gamma ** tick * reward
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert discounted == pytest.approx(initial)
    np.testing.assert_array_equal(reference.q_values(obs), np.zeros(4))


def test_observation_mapping_controls_action_labels_even_with_default_config():
    env = configured(position=(0, 1), goal=(0, 0))
    reference = VisibleOptimalQ(env.config)
    ordinary = env.observe()
    remapped = ordinary.copy()
    labels = [3, 2, 0, 1]
    remapped[-16:] = np.eye(4, dtype=np.float32)[labels].ravel()
    np.testing.assert_allclose(reference.q_values(remapped), reference.q_values(ordinary)[labels])
    assert reference.act(ordinary) == 2
    assert reference.act(remapped) == 1
    assert len(reference._cache) == 1


def test_action_finish_mask_identifies_the_last_salvageable_action():
    env = configured(position=(0, 2), goal=(0, 0), horizon=2)
    reference = VisibleOptimalQ(env.config)
    assert reference.can_finish(env.observe())
    np.testing.assert_array_equal(reference.action_can_finish(env.observe()), [False, False, True, False])
    remapped = env.observe().copy()
    remapped[-16:] = np.eye(4, dtype=np.float32)[[3, 2, 0, 1]].ravel()
    np.testing.assert_array_equal(reference.action_can_finish(remapped), [False, True, False, False])
    # Wasting a tick loses the remaining chance; all future actions are tied
    # on success reachability, which must not be called an avoidable mistake.
    observation, _, _, _, _ = env.step(0)
    assert not reference.can_finish(observation)
    assert not reference.action_can_finish(observation).any()


def test_rejects_unsupported_or_invalid_visible_state_and_bounds_cache():
    with pytest.raises(ValueError, match='exactly one pellet'):
        VisibleOptimalQ(WorldConfig(pellet_count=2))
    env = configured()
    reference = VisibleOptimalQ(env.config, max_cached_maps=2)
    obs = env.observe()
    invalid = obs.copy()
    invalid[25 + 1] = 1
    with pytest.raises(ValueError, match='at most one'):
        reference.q_values(invalid)
    invalid = obs.copy()
    invalid[-16:] = 0
    with pytest.raises(ValueError, match='permutation'):
        reference.q_values(invalid)
    for seed in range(5):
        sample = CollectionWorld(env.config)
        sample.reset(seed=seed)
        reference.q_values(sample.observe())
    assert len(reference._cache) == 2
