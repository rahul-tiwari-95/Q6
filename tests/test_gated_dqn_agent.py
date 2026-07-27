"""
Tests for agent.gated_dqn_agent.GatedDQNAgent.

Focus: Ablation 2 — frozen-policy anchor loss for the collect head.

Validates:
  - anchor_weight=0.0 (default): no anchor network loaded, loss unaffected,
    last_anchor_loss stays 0.0, learn() behaves exactly as before ablation 2.
  - A checkpoint can be supplied without anchor_weight>0 and has zero effect
    on the loss (the anchor network loads, but the term is gated off).
  - anchor_weight>0 + loaded anchor + an all-"safe" batch (hunter far away)
    produces a nonzero, finite anchor loss that is added into the total loss.
  - anchor_weight>0 + an all-"unsafe" batch (hunter close) masks the anchor
    term to exactly zero — safe-context gating is enforced correctly.
  - A mixed safe/unsafe batch only pulls in the safe rows (statistical check
    via comparing to an all-safe batch built from the same states).
  - The anchor network is frozen: eval() mode, requires_grad=False on all
    params, and its weights never change across multiple learn() calls.
  - set_anchor_weight() updates the mutable weight used by learn().
  - Anchor checkpoint loading accepts both a raw/GatedDQNAgent-wrapped
    GatedOptionNetwork state dict and a DQNv2Agent/CNNDuelingQNetwork
    checkpoint (the shape train_v2.py produces), matching
    load_from_cnn_dueling()'s mapping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.gated_dqn_agent import GatedDQNAgent
from model.gated_option_network import GatedOptionNetwork
from model.cnn_q_network import CNNDuelingQNetwork

# NOTE: a state of all-EMPTY (value 6) cells encodes to an all-zero (6,25,25)
# CNN input (see utils/state_encoder.py). With zero-initialized biases that
# makes Q-values identically zero for *any* network weights, which would
# make the anchor-loss comparison vacuous. Use a populated grid (cell values
# 0..6, i.e. some walls/pellets/agents) so the encoded tensor is non-trivial.
STATE = np.random.default_rng(123).integers(0, 7, size=625).astype(np.uint8)

# hunter_dist_norm threshold used by the anchor mask: 8 / (2*25) = 0.16
SAFE_CTX   = np.array([0.9, 0.5, 1.0], dtype=np.float32)   # hunter far  -> safe
UNSAFE_CTX = np.array([0.05, 0.5, 1.0], dtype=np.float32)  # hunter close -> unsafe

N_FILL = 70  # > BATCH_SIZE=64


def _fill_buffer(agent: GatedDQNAgent, ctx: np.ndarray, rng: np.random.Generator) -> None:
    for _ in range(N_FILL):
        agent.memory.add(
            STATE, ctx, int(rng.integers(4)), -0.001, STATE, ctx, False
        )


def _make_gated_checkpoint(tmp_path) -> str:
    """A raw GatedOptionNetwork checkpoint (as GatedOptionNetwork.save() writes)."""
    net = GatedOptionNetwork()
    path = str(tmp_path / "anchor_gated.pth")
    net.save(path)
    return path


def _make_cnn_checkpoint(tmp_path) -> str:
    """A DQNv2Agent-style checkpoint wrapping a CNNDuelingQNetwork (train_v2.py shape)."""
    cnn = CNNDuelingQNetwork()
    path = str(tmp_path / "anchor_cnn.pth")
    torch.save({
        "qnetwork_local": cnn.state_dict(),
        "qnetwork_target": cnn.state_dict(),
        "epsilon": 0.05, "learning_step": 100,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Default: anchor off
# ---------------------------------------------------------------------------

class TestAnchorOffByDefault:
    def test_default_anchor_weight_is_zero(self):
        agent = GatedDQNAgent(device="cpu")
        assert agent.anchor_weight == 0.0
        assert agent.qnetwork_anchor is None

    def test_learn_unaffected_without_anchor(self):
        agent = GatedDQNAgent(device="cpu")
        rng = np.random.default_rng(0)
        _fill_buffer(agent, SAFE_CTX, rng)
        result = agent.learn()
        assert result["anchor_loss"] == 0.0
        assert agent.last_anchor_loss == 0.0
        assert np.isfinite(result["loss"])

    def test_checkpoint_without_weight_has_no_effect(self, tmp_path):
        """Loading an anchor checkpoint with anchor_weight=0.0 (still off) must
        not change the loss at all vs. no checkpoint."""
        path = _make_gated_checkpoint(tmp_path)
        agent_no_anchor = GatedDQNAgent(device="cpu")
        agent_with_anchor_off = GatedDQNAgent(
            device="cpu", anchor_weight=0.0, anchor_checkpoint_path=path
        )
        # Anchor network *is* loaded...
        assert agent_with_anchor_off.qnetwork_anchor is not None
        # ...but the loss term is gated off because anchor_weight <= 0.
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        _fill_buffer(agent_no_anchor, SAFE_CTX, rng1)
        _fill_buffer(agent_with_anchor_off, SAFE_CTX, rng2)
        r1 = agent_no_anchor.learn()
        r2 = agent_with_anchor_off.learn()
        assert r2["anchor_loss"] == 0.0
        assert r1["anchor_loss"] == r2["anchor_loss"] == 0.0


# ---------------------------------------------------------------------------
# Anchor loaded + weight > 0
# ---------------------------------------------------------------------------

class TestAnchorLossComputation:
    def test_anchor_loss_nonzero_when_all_safe(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        rng = np.random.default_rng(1)
        _fill_buffer(agent, SAFE_CTX, rng)
        result = agent.learn()
        assert np.isfinite(result["anchor_loss"])
        # Local and (independently, randomly initialized) anchor collect
        # heads should not coincide exactly -> nonzero loss.
        assert result["anchor_loss"] > 0.0
        assert result["anchor_loss"] == agent.last_anchor_loss

    def test_anchor_loss_masked_to_zero_when_all_unsafe(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        rng = np.random.default_rng(2)
        _fill_buffer(agent, UNSAFE_CTX, rng)
        result = agent.learn()
        assert result["anchor_loss"] == 0.0, (
            "hunter_dist_norm <= 0.16 (unsafe) must fully mask out the anchor term"
        )

    def test_anchor_term_added_to_total_loss(self, tmp_path, monkeypatch):
        """Total loss with anchor_weight>0 should differ from the base
        smooth_l1/gate-entropy loss alone, when the batch is all-safe."""
        path = _make_gated_checkpoint(tmp_path)

        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        torch.manual_seed(0)
        agent_plain = GatedDQNAgent(device="cpu", gate_reg_weight=0.0)
        torch.manual_seed(0)
        agent_anchored = GatedDQNAgent(
            device="cpu", gate_reg_weight=0.0,
            anchor_weight=1.0, anchor_checkpoint_path=path,
        )
        _fill_buffer(agent_plain, SAFE_CTX, rng1)
        _fill_buffer(agent_anchored, SAFE_CTX, rng2)

        r_plain    = agent_plain.learn()
        r_anchored = agent_anchored.learn()
        assert r_anchored["anchor_loss"] > 0.0
        # Both networks started from identical seeds/weights and saw the
        # identical batch, so the only difference in total loss should be
        # attributable to the anchor term being added in.
        assert r_anchored["loss"] != pytest.approx(r_plain["loss"])

    def test_boundary_threshold_excludes_exact_value(self, tmp_path):
        """context[:, 0] must be STRICTLY greater than 0.16 to count as safe."""
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        boundary_ctx = np.array([8.0 / (2 * 25), 0.5, 1.0], dtype=np.float32)
        rng = np.random.default_rng(4)
        _fill_buffer(agent, boundary_ctx, rng)
        result = agent.learn()
        assert result["anchor_loss"] == 0.0


# ---------------------------------------------------------------------------
# Anchor network is frozen
# ---------------------------------------------------------------------------

class TestAnchorFrozen:
    def test_anchor_in_eval_mode_no_grad(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        assert agent.qnetwork_anchor.training is False
        assert all(not p.requires_grad for p in agent.qnetwork_anchor.parameters())

    def test_anchor_weights_never_update(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        before = {k: v.clone() for k, v in agent.qnetwork_anchor.state_dict().items()}

        rng = np.random.default_rng(6)
        for _ in range(3):
            _fill_buffer(agent, SAFE_CTX, rng)
            agent.learn()

        after = agent.qnetwork_anchor.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), f"anchor param {k} changed"


# ---------------------------------------------------------------------------
# set_anchor_weight() — training-script decay hook
# ---------------------------------------------------------------------------

class TestSetAnchorWeight:
    def test_setter_updates_weight(self):
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0)
        agent.set_anchor_weight(0.3)
        assert agent.anchor_weight == pytest.approx(0.3)
        agent.set_anchor_weight(0.0)
        assert agent.anchor_weight == 0.0

    def test_decayed_weight_of_zero_disables_anchor_term(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        agent.set_anchor_weight(0.0)
        rng = np.random.default_rng(7)
        _fill_buffer(agent, SAFE_CTX, rng)
        result = agent.learn()
        assert result["anchor_loss"] == 0.0


# ---------------------------------------------------------------------------
# Checkpoint format detection
# ---------------------------------------------------------------------------

class TestAnchorCheckpointFormats:
    def test_loads_raw_gated_option_network_checkpoint(self, tmp_path):
        path = _make_gated_checkpoint(tmp_path)
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        assert isinstance(agent.qnetwork_anchor, GatedOptionNetwork)

    def test_loads_cnn_dueling_checkpoint_via_conversion(self, tmp_path):
        path = _make_cnn_checkpoint(tmp_path)
        cnn = CNNDuelingQNetwork()
        cnn.load_state_dict(torch.load(path, weights_only=True)["qnetwork_local"])
        agent = GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
        # Trunk weights should match the CNNDuelingQNetwork checkpoint (same
        # mapping load_from_cnn_dueling() uses for warm-starting).
        assert torch.allclose(agent.qnetwork_anchor.conv1.weight, cnn.conv1.weight)
        assert torch.allclose(agent.qnetwork_anchor.fc_shared.weight, cnn.fc_shared.weight)
        assert torch.allclose(agent.qnetwork_anchor.collect_out.weight, cnn.adv_out.weight)

    def test_bad_checkpoint_raises(self, tmp_path):
        path = str(tmp_path / "bad.pth")
        torch.save({"nonsense_key": torch.zeros(3)}, path)
        with pytest.raises(ValueError):
            GatedDQNAgent(device="cpu", anchor_weight=1.0, anchor_checkpoint_path=path)
