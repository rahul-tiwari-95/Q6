"""
Tests for model/cnn_q_network.py — CNN + Dueling Q-network.

Tests target structural invariants (shapes, dueling math identity, gradient
flow, save/load) rather than fixed numeric outputs.
"""

import numpy as np
import pytest
import torch

from model.cnn_q_network import CNNDuelingQNetwork


# ----------------------------- shapes -----------------------------

class TestShapes:
    def test_single_input_shape(self):
        net = CNNDuelingQNetwork()
        x = torch.randn(1, 6, 25, 25)
        out = net(x)
        assert out.shape == (1, 4)

    def test_batch_shape(self):
        net = CNNDuelingQNetwork()
        x = torch.randn(32, 6, 25, 25)
        out = net(x)
        assert out.shape == (32, 4)

    def test_finite_outputs(self):
        net = CNNDuelingQNetwork()
        x = torch.randn(8, 6, 25, 25)
        out = net(x)
        assert torch.all(torch.isfinite(out))

    def test_rejects_wrong_input_dim(self):
        net = CNNDuelingQNetwork()
        with pytest.raises(ValueError):
            net(torch.randn(6, 25, 25))  # missing batch dim


# ----------------------------- dueling identity -----------------------------

class TestDuelingMath:
    def test_advantage_mean_zero_after_centering(self):
        """For dueling with mean baseline: Q - mean(Q) over actions equals A - mean(A).
        We can verify by hooking the network and recomputing internally, but a
        simpler proxy: if we add a constant to V it shifts all Qs equally."""
        torch.manual_seed(0)
        net = CNNDuelingQNetwork()
        x = torch.randn(4, 6, 25, 25)
        with torch.no_grad():
            q1 = net(x)
            # Add a constant to V's bias and recheck — all Qs should shift identically.
            net.value_out.bias.add_(1.5)
            q2 = net(x)
        diff = q2 - q1
        # Every entry should equal 1.5 (the shift), within float tolerance.
        assert torch.allclose(diff, torch.full_like(diff, 1.5), atol=1e-5)

    def test_advantage_shift_invariance(self):
        """Adding a constant to ADV bias should NOT change Q (mean-centered)."""
        torch.manual_seed(0)
        net = CNNDuelingQNetwork()
        x = torch.randn(4, 6, 25, 25)
        with torch.no_grad():
            q1 = net(x)
            net.adv_out.bias.add_(2.3)
            q2 = net(x)
        assert torch.allclose(q1, q2, atol=1e-5)


# ----------------------------- learning capacity -----------------------------

class TestGradientFlow:
    def test_loss_backward_produces_grads_everywhere(self):
        net = CNNDuelingQNetwork()
        x = torch.randn(8, 6, 25, 25)
        target = torch.randn(8, 4)
        out = net(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        # Every parameter must have a non-None, finite gradient
        for name, p in net.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert torch.all(torch.isfinite(p.grad)), f"{name} has non-finite grad"

    def test_can_overfit_tiny_batch(self):
        """Sanity: net has enough capacity to fit a tiny batch in a few steps."""
        torch.manual_seed(0)
        net = CNNDuelingQNetwork()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.randn(4, 6, 25, 25)
        target = torch.randn(4, 4)
        initial_loss = ((net(x) - target) ** 2).mean().item()
        for _ in range(200):
            opt.zero_grad()
            loss = ((net(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, \
            f"net failed to overfit tiny batch: {initial_loss=:.4f} -> {final_loss=:.4f}"


# ----------------------------- persistence -----------------------------

class TestPersistence:
    def test_save_load_preserves_outputs(self, tmp_path):
        torch.manual_seed(42)
        net1 = CNNDuelingQNetwork()
        x = torch.randn(3, 6, 25, 25)
        net1.eval()
        with torch.no_grad():
            out1 = net1(x).clone()
        path = str(tmp_path / "cnn.pth")
        net1.save(path)

        net2 = CNNDuelingQNetwork()
        net2.load(path)
        net2.eval()
        with torch.no_grad():
            out2 = net2(x)
        torch.testing.assert_close(out1, out2)


# ----------------------------- integration -----------------------------

class TestEnvIntegration:
    def test_accepts_encoded_env_state(self):
        from environment.hunter_gridworld import HunterGridworld
        from utils.state_encoder import encode_state
        env = HunterGridworld(difficulty_level=4, curriculum_phase=4)
        state, _ = env.reset(seed=0)
        encoded = encode_state(state)  # (6, 25, 25)
        tensor = torch.from_numpy(encoded).unsqueeze(0)  # (1, 6, 25, 25)
        net = CNNDuelingQNetwork()
        out = net(tensor)
        assert out.shape == (1, 4)
        assert torch.all(torch.isfinite(out))
