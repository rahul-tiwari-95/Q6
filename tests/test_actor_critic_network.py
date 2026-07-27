"""
Tests for model/actor_critic_network.py — CNN actor-critic used by PPOAgent (v8).
"""

import torch

from config import GRID_SIZE
from model.actor_critic_network import CNNActorCritic
from utils.state_encoder import NUM_CHANNELS


def _random_batch(batch_size: int = 8) -> torch.Tensor:
    return torch.rand(batch_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)


class TestForward:
    def test_output_shapes(self):
        net = CNNActorCritic(in_channels=NUM_CHANNELS, grid_size=GRID_SIZE, n_actions=4)
        logits, value = net(_random_batch(8))
        assert logits.shape == (8, 4)
        assert value.shape == (8,)

    def test_rejects_wrong_rank(self):
        net = CNNActorCritic()
        try:
            net(torch.rand(NUM_CHANNELS, GRID_SIZE, GRID_SIZE))  # missing batch dim
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_batch_size_one(self):
        net = CNNActorCritic()
        logits, value = net(_random_batch(1))
        assert logits.shape == (1, 4)
        assert value.shape == (1,)


class TestGetActionAndValue:
    def test_sampled_action_in_range(self):
        net = CNNActorCritic(n_actions=4)
        action, logprob, entropy, value = net.get_action_and_value(_random_batch(16))
        assert action.shape == (16,)
        assert int(action.min()) >= 0 and int(action.max()) < 4
        assert logprob.shape == (16,)
        assert entropy.shape == (16,)
        assert value.shape == (16,)
        assert torch.isfinite(logprob).all()
        assert (entropy >= 0).all()  # categorical entropy is non-negative

    def test_given_action_is_reused_not_resampled(self):
        net = CNNActorCritic(n_actions=4)
        x = _random_batch(5)
        forced_action = torch.tensor([0, 1, 2, 3, 0])
        action, logprob, _entropy, _value = net.get_action_and_value(x, action=forced_action)
        assert torch.equal(action, forced_action)
        # log_prob of the forced action must match a manual softmax lookup
        logits, _ = net(x)
        log_probs_manual = torch.log_softmax(logits, dim=-1)
        expected = log_probs_manual.gather(1, forced_action.unsqueeze(1)).squeeze(1)
        assert torch.allclose(logprob, expected, atol=1e-5)

    def test_uniform_init_near_max_entropy(self):
        # Policy head is std=0.01 orthogonal init -> logits near zero ->
        # near-uniform categorical -> entropy close to log(n_actions).
        net = CNNActorCritic(n_actions=4)
        _action, _logprob, entropy, _value = net.get_action_and_value(_random_batch(32))
        assert entropy.mean().item() > 1.2  # log(4) ~= 1.386; allow init noise


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        net = CNNActorCritic()
        path = tmp_path / "net.pth"
        net.save(str(path))

        net2 = CNNActorCritic()
        net2.load(str(path))

        x = _random_batch(4)
        l1, v1 = net(x)
        l2, v2 = net2(x)
        assert torch.allclose(l1, l2)
        assert torch.allclose(v1, v2)
