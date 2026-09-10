"""Evaluation isolation and exact learning-state continuation checks."""

import copy
import json

import numpy as np
import torch
import pytest

from q6.adaptation import evaluate, run_study
from q6.learning import DQN
from q6.world import CollectionWorld, WorldConfig


def test_evaluation_does_not_change_training_state_or_future_random_actions():
    torch.set_num_threads(1)
    env = CollectionWorld()
    observation, _ = env.reset(seed=8)
    agent = DQN(env.observation_size, seed=4, capacity=512)
    untouched = DQN.from_state(agent.state_dict())
    before = agent.parameter_hash()
    rng_before = copy.deepcopy(agent.rng.bit_generator.state)
    rows, trajectory = evaluate(agent, env.config, [900000, 900001], seed=4,
                                condition="continued", checkpoint="after_a", world="A")
    assert agent.parameter_hash() == before
    assert agent.steps == agent.updates == agent.count == 0
    assert agent.rng.bit_generator.state == rng_before
    assert agent.optimizer.state_dict() == untouched.optimizer.state_dict()
    assert trajectory["map_seed"] == 900000 and len(rows) == 2
    assert [agent.act(observation, 1) for _ in range(20)] == [untouched.act(observation, 1) for _ in range(20)]
    repeated, _ = evaluate(agent, env.config, [900000, 900001], seed=4,
                           condition="continued", checkpoint="after_a", world="A")
    assert repeated == rows


def test_learning_checkpoint_preserves_adam_replay_rng_and_update_cadence():
    torch.set_num_threads(1)
    env = CollectionWorld()
    observation, _ = env.reset(seed=0)
    agent = DQN(env.observation_size, seed=1, capacity=512)
    for index in range(260):
        agent.observe(observation, index % 4, 0.1, observation, index % 7 == 0)
    clone = DQN.from_state(agent.state_dict())
    assert agent.updates > 0
    for _ in range(8):
        agent.observe(observation, 2, 1.0, observation, True)
        clone.observe(observation, 2, 1.0, observation, True)
    assert agent.parameter_hash() == clone.parameter_hash()
    assert agent.updates == clone.updates and agent.last_loss == clone.last_loss
    clone.states[0] = 7
    assert not np.array_equal(agent.states[0], clone.states[0])


@pytest.mark.parametrize("ended, expected_loss", [(True, 0.5), (False, 10.2)])
def test_bellman_targets_use_executed_action_double_selection_and_terminal_mask(ended, expected_loss):
    # Online chooses action3; target's own maximum is action0. The correct
    # nonterminal target is 1 + .97*10, not 1 + .97*100. A terminal must use1.
    agent = DQN(8, seed=2, capacity=256)
    with torch.no_grad():
        for network in (agent.online, agent.target):
            for parameter in network.parameters():
                parameter.zero_()
        agent.online[-1].bias.copy_(torch.tensor([0., 1., 2., 3.]))
        agent.target[-1].bias.copy_(torch.tensor([100., 20., 30., 10.]))
    agent.count = 64
    agent.states[:64] = agent.next_states[:64] = 0
    agent.actions[:64] = 0
    agent.rewards[:64] = 1
    agent.ends[:64] = ended
    previous_hash = agent.parameter_hash()
    agent.learn()
    assert agent.last_loss == pytest.approx(expected_loss)
    assert agent.parameter_hash() != previous_hash


@pytest.mark.parametrize("kwargs", [
    {"capacity": 128},
    {"capacity": 256, "batch_size": 512},
    {"batch_size": 0},
    {"batch_size": -1},
    {"observation_size": 0},
])
def test_invalid_replay_and_observation_configurations_fail_explicitly(kwargs):
    options = {"observation_size": 8, **kwargs}
    with pytest.raises(ValueError):
        DQN(**options)


def test_smoke_study_preserves_panel_and_frozen_parameters(tmp_path):
    result = run_study(tmp_path / "smoke", [0], phase_steps=16, eval_episodes=2, max_seconds=30)
    assert result["run"]["status"] == "complete"
    assert result["run"]["training_steps"] == 48
    assert result["protocol"]["rule_visibility"] == "observed"
    frozen = [r for r in result["runs"] if r["condition"] == "frozen_after_a"]
    assert len({r["parameter_hash"] for r in frozen}) == 1
    assert {r["train_steps"] for r in frozen} == {16}
    for world in ("A", "B"):
        assert len({r["success_rate"] for r in frozen if r["world"] == world}) == 1
    assert {t["map_seed"] for t in result["trajectories"]} == {900000}
    saved = json.loads((tmp_path / "smoke" / "results.json").read_text())
    assert saved["run"]["exploratory"] is True


def test_new_panel_and_protocol_are_recorded_without_training_overlap(tmp_path):
    protocol = tmp_path / "adaptation_protocol_v2.md"
    protocol.write_text("A prospective small test protocol.\n")
    result = run_study(tmp_path / "new_panel", [0], phase_steps=4, eval_episodes=1,
                       max_seconds=30, eval_seed_start=910000, protocol_file=protocol)
    assert result["protocol"]["id"] == "adaptation-v2"
    assert result["protocol"]["eval_panel"] == [910000]
    assert {t["map_seed"] for t in result["trajectories"]} == {910000}
    assert result["run"]["model_parameter_count"] > 0
    assert (tmp_path / "new_panel/source/q6/world.py").exists()
    assert (tmp_path / "new_panel/protocol.md").read_text() == protocol.read_text()
