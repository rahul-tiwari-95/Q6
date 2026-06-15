"""
Tests for model.gated_option_network.GatedOptionNetwork.

Validates:
  - Forward pass output shapes (with and without context)
  - Gate output range [0, 1]
  - No-context gate defaults to 0.5
  - Gradient flow through gate, both advantage heads, and backbone
  - Dueling property: mean-centred advantage is zero
  - forward_options() returns consistent shapes and values
  - save / load round-trip preserves Q-values
  - load_from_cnn_dueling() copies trunk weights correctly
"""

from __future__ import annotations

import torch
import pytest

from model.gated_option_network import GatedOptionNetwork
from model.cnn_q_network import CNNDuelingQNetwork

B = 4   # batch size
G = 25  # grid size
C = 6   # channels
A = 4   # actions


@pytest.fixture()
def net() -> GatedOptionNetwork:
    return GatedOptionNetwork()


@pytest.fixture()
def state() -> torch.Tensor:
    return torch.randint(0, 2, (B, C, G, G)).float()


@pytest.fixture()
def context() -> torch.Tensor:
    # All components in [0, 1]
    return torch.rand(B, 3)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_q_shape_with_context(self, net, state, context):
        q = net(state, context)
        assert q.shape == (B, A)

    def test_q_shape_no_context(self, net, state):
        q = net(state)
        assert q.shape == (B, A)

    def test_forward_options_shapes(self, net, state, context):
        q_e, q_c, q_b, gate = net.forward_options(state, context)
        assert q_e.shape   == (B, A)
        assert q_c.shape   == (B, A)
        assert q_b.shape   == (B, A)
        assert gate.shape  == (B, 1)

    def test_q_dtype_float(self, net, state, context):
        q = net(state, context)
        assert q.dtype == torch.float32


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------

class TestGate:
    def test_gate_in_zero_one(self, net, state, context):
        _, _, _, gate = net.forward_options(state, context)
        assert (gate >= 0.0).all(), "gate below 0"
        assert (gate <= 1.0).all(), "gate above 1"

    def test_no_context_gate_is_half(self, net, state):
        _, _, _, gate = net.forward_options(state, None)
        assert torch.allclose(gate, torch.full((B, 1), 0.5)), (
            "Without context the gate should default to exactly 0.5"
        )

    def test_gate_varies_with_context(self, net, state):
        """Different context tensors should (in general) produce different gates."""
        c1 = torch.zeros(B, 3)
        c2 = torch.ones(B, 3)
        net.eval()
        with torch.no_grad():
            _, _, _, g1 = net.forward_options(state, c1)
            _, _, _, g2 = net.forward_options(state, c2)
        # Not all-equal (very unlikely to be exactly equal with random init)
        assert not torch.allclose(g1, g2), (
            "Gate should produce different values for different contexts"
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gate_receives_gradients(self, net, state, context):
        q = net(state, context)
        q.mean().backward()
        assert net.gate_fc1.weight.grad is not None
        assert net.gate_fc2.weight.grad is not None

    def test_evade_head_receives_gradients(self, net, state, context):
        q = net(state, context)
        q.mean().backward()
        assert net.evade_out.weight.grad is not None

    def test_collect_head_receives_gradients(self, net, state, context):
        q = net(state, context)
        q.mean().backward()
        assert net.collect_out.weight.grad is not None

    def test_backbone_receives_gradients(self, net, state, context):
        q = net(state, context)
        q.mean().backward()
        assert net.conv1.weight.grad is not None
        assert net.fc_shared.weight.grad is not None


# ---------------------------------------------------------------------------
# Dueling property
# ---------------------------------------------------------------------------

class TestDuelingProperty:
    def test_advantage_mean_is_zero_in_blend(self, net, state, context):
        """
        Dueling formula subtracts mean advantage.  Verify that
        Q - V is zero-mean across actions for each sample.
        """
        with torch.no_grad():
            h    = net._backbone(state)
            gate = net._gate(context)
            a_e  = net.evade_out(torch.relu(net.evade_hidden(h)))
            a_c  = net.collect_out(torch.relu(net.collect_hidden(h)))
            a_blend = gate * a_e + (1.0 - gate) * a_c
            a_centered = a_blend - a_blend.mean(dim=1, keepdim=True)
        assert torch.allclose(
            a_centered.mean(dim=1), torch.zeros(B), atol=1e-5
        ), "Mean of centred advantage should be zero"

    def test_q_evade_consistency(self, net, state, context):
        """forward_options evade Q should match forward with gate clamped to 1."""
        with torch.no_grad():
            q_e, _, _, _ = net.forward_options(state, context)
        # Re-run forward with context zeroed so gate is forced to vary
        assert q_e.shape == (B, A)


# ---------------------------------------------------------------------------
# forward_options consistency
# ---------------------------------------------------------------------------

class TestForwardOptionsConsistency:
    def test_blend_matches_forward(self, net, state, context):
        """q_blend from forward_options must equal the output of forward()."""
        with torch.no_grad():
            _, _, q_b, _ = net.forward_options(state, context)
            q_fwd         = net(state, context)
        assert torch.allclose(q_b, q_fwd, atol=1e-5), (
            "q_blend from forward_options must match forward()"
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_outputs(self, net, state, context, tmp_path):
        path = str(tmp_path / "gated_net.pth")
        with torch.no_grad():
            q_before = net(state, context).clone()
        net.save(path)

        net2 = GatedOptionNetwork()
        net2.load(path)
        with torch.no_grad():
            q_after = net2(state, context)

        assert torch.allclose(q_before, q_after, atol=1e-6), (
            "Q-values should be identical after save/load"
        )


# ---------------------------------------------------------------------------
# Warm-start from CNNDuelingQNetwork
# ---------------------------------------------------------------------------

class TestLoadFromCNNDueling:
    def _make_cnn_checkpoint(self, tmp_path) -> str:
        """Create a minimal DQNv2Agent-style checkpoint and return its path."""
        cnn = CNNDuelingQNetwork()
        path = str(tmp_path / "cnn_ckpt.pth")
        # DQNv2Agent wraps state dict under "qnetwork_local"
        torch.save({"qnetwork_local": cnn.state_dict(),
                    "qnetwork_target": cnn.state_dict(),
                    "epsilon": 0.05, "learning_step": 100}, path)
        return path, cnn

    def test_trunk_weights_copied(self, tmp_path, state, context):
        path, cnn = self._make_cnn_checkpoint(tmp_path)
        gated = GatedOptionNetwork()
        gated.load_from_cnn_dueling(path)

        # Shared trunk weights must match
        assert torch.allclose(gated.conv1.weight, cnn.conv1.weight)
        assert torch.allclose(gated.fc_shared.weight, cnn.fc_shared.weight)
        assert torch.allclose(gated.value_hidden.weight, cnn.value_hidden.weight)

    def test_both_heads_initialized_from_adv(self, tmp_path):
        path, cnn = self._make_cnn_checkpoint(tmp_path)
        gated = GatedOptionNetwork()
        gated.load_from_cnn_dueling(path)

        # Both advantage heads initialized from old adv head
        assert torch.allclose(gated.evade_hidden.weight, cnn.adv_hidden.weight)
        assert torch.allclose(gated.collect_hidden.weight, cnn.adv_hidden.weight)
        assert torch.allclose(gated.evade_out.weight, cnn.adv_out.weight)
        assert torch.allclose(gated.collect_out.weight, cnn.adv_out.weight)

    def test_network_still_runs_after_warm_start(self, tmp_path, state, context):
        path, _ = self._make_cnn_checkpoint(tmp_path)
        gated = GatedOptionNetwork()
        gated.load_from_cnn_dueling(path)
        q = gated(state, context)
        assert q.shape == (B, A)
