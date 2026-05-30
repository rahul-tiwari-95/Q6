"""
Tests for model/q_network.py - Q-Network architecture.
"""
import pytest
import torch
import numpy as np
from model.q_network import QNetwork


class TestQNetwork:
    """Test Q-Network architecture and behavior."""

    def test_default_architecture(self):
        net = QNetwork()
        x = torch.randn(1, 625)
        output = net(x)
        assert output.shape == (1, 4)  # 4 actions

    def test_custom_hidden_sizes(self):
        net = QNetwork(input_size=625, hidden_sizes=(512, 256, 128), output_size=4)
        x = torch.randn(2, 625)
        output = net(x)
        assert output.shape == (2, 4)

    def test_batch_processing(self):
        net = QNetwork()
        x = torch.randn(32, 625)  # Batch of 32
        output = net(x)
        assert output.shape == (32, 4)
        assert len(output[0]) == 4  # Each has Q-values for 4 actions

    def test_output_bounded(self):
        """Q-values should be finite (no overflow)."""
        net = QNetwork()
        x = torch.randn(10, 625)
        output = net(x)
        assert torch.all(torch.isfinite(output))

    def test_save_load(self, tmp_path):
        net1 = QNetwork()
        with torch.no_grad():
            orig_vals = net1(torch.randn(1, 625)).clone()
        path = str(tmp_path / "test_model.pth")
        net1.save(path)

        net2 = QNetwork()
        net2.load(path)
        with torch.no_grad():
            loaded_vals = net2(torch.randn(1, 625))
        # Outputs use same weights now
        torch.testing.assert_close(net1.state_dict()['layers.0.weight'], net2.state_dict()['layers.0.weight'])

    def test_reproducibility_with_same_weights(self):
        net1 = QNetwork()
        net1.eval()
        x = torch.randn(1, 625)
        out1 = net1(x)
        out2 = net1(x)
        torch.testing.assert_close(out1, out2)

    def test_uses_config_constants(self):
        """Q-Network should work with config constants."""
        from config import STATE_SIZE, ACTION_SIZE, HIDDEN_SIZES
        net = QNetwork(STATE_SIZE, HIDDEN_SIZES, ACTION_SIZE)
        x = torch.randn(1, STATE_SIZE)
        output = net(x)
        assert output.shape == (1, ACTION_SIZE)
