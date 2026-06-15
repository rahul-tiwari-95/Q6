"""
Gated Option Q-Network — Dual-head architecture with learned context gate.

Architecture
------------
Input:   (B, 6, 25, 25) float tensor  (channel-encoded state)
Context: (B, 3) float tensor          [hunter_dist_norm, pellets_remaining_norm, lives_norm]

    Shared CNN backbone:
        Conv2d(6 → 32,  k=3, p=1) → ReLU
        Conv2d(32 → 64, k=3, p=1) → ReLU
        Flatten → Linear(64*25*25 → 256) → ReLU

    Two specialized advantage heads:
        A_evade(s,a):   Linear(256 → 128) → ReLU → Linear(128 → n_actions)
        A_collect(s,a): Linear(256 → 128) → ReLU → Linear(128 → n_actions)

    Shared value head (Dueling decomposition):
        V(s):           Linear(256 → 128) → ReLU → Linear(128 → 1)

    Gate MLP (context → [0, 1] blend scalar):
        Linear(3 → 16) → ReLU → Linear(16 → 1) → Sigmoid
        gate = 1.0  →  pure evasion
        gate = 0.0  →  pure collection

    Blended Q (Dueling + Gating):
        A_blend = gate * A_evade + (1 − gate) * A_collect
        Q(s,a)  = V(s) + (A_blend(s,a) − mean_a A_blend(s,a))

The gate is trained end-to-end by gradient descent through the Q-learning
loss.  Over time it learns that high hunter proximity should activate evasion
mode and low hunter pressure should activate collection mode.

`forward_options()` exposes evade-only and collect-only Q-values as well as
the raw gate scalar — used in the dashboard Brain tab for visualization.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedOptionNetwork(nn.Module):
    CONTEXT_DIM: int = 3  # [hunter_dist_norm, pellets_remaining_norm, lives_norm]

    def __init__(
        self,
        in_channels: int = 6,
        grid_size: int = 25,
        n_actions: int = 4,
        conv_channels: Tuple[int, int] = (32, 64),
        fc_hidden: int = 256,
        head_hidden: int = 128,
        gate_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.n_actions = n_actions

        c1, c2 = conv_channels

        # Shared CNN backbone
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        flat_size = c2 * grid_size * grid_size
        self.fc_shared = nn.Linear(flat_size, fc_hidden)

        # Evasion advantage head
        self.evade_hidden = nn.Linear(fc_hidden, head_hidden)
        self.evade_out = nn.Linear(head_hidden, n_actions)

        # Collection advantage head
        self.collect_hidden = nn.Linear(fc_hidden, head_hidden)
        self.collect_out = nn.Linear(head_hidden, n_actions)

        # Shared value head (Dueling)
        self.value_hidden = nn.Linear(fc_hidden, head_hidden)
        self.value_out = nn.Linear(head_hidden, 1)

        # Gate MLP (context-conditioned blend)
        self.gate_fc1 = nn.Linear(self.CONTEXT_DIM, gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, 1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    # Sub-modules exposed for warm-start weight mapping
    # ------------------------------------------------------------------
    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """CNN trunk: (B, C, H, W) → (B, fc_hidden)."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        return F.relu(self.fc_shared(x))

    def _gate(self, context: torch.Tensor) -> torch.Tensor:
        """Gate MLP: (B, 3) → (B, 1) in [0, 1]."""
        g = F.relu(self.gate_fc1(context.float()))
        return torch.sigmoid(self.gate_fc2(g))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, 6, 25, 25) channel-encoded state.
            context: (B, 3) gate context — [hunter_dist_norm, pellets_norm, lives_norm].
                     If None the gate defaults to 0.5 (balanced blend).
        Returns:
            Q: (B, n_actions) blended Q-values.
        """
        h = self._backbone(x)

        if context is not None:
            gate = self._gate(context)                              # (B, 1)
        else:
            gate = torch.full((x.size(0), 1), 0.5,
                              device=x.device, dtype=torch.float32)

        a_evade   = self.evade_out(F.relu(self.evade_hidden(h)))    # (B, n_actions)
        a_collect = self.collect_out(F.relu(self.collect_hidden(h))) # (B, n_actions)
        a_blend   = gate * a_evade + (1.0 - gate) * a_collect

        v = self.value_out(F.relu(self.value_hidden(h)))            # (B, 1)
        q = v + (a_blend - a_blend.mean(dim=1, keepdim=True))
        return q

    def forward_options(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exposes all intermediate values for visualization and analysis.

        Returns:
            q_evade:   (B, n_actions) — Q-values using evade head only (gate=1).
            q_collect: (B, n_actions) — Q-values using collect head only (gate=0).
            q_blend:   (B, n_actions) — gated blend (same as forward()).
            gate:      (B, 1)         — gate scalar in [0, 1].
        """
        h = self._backbone(x)

        if context is not None:
            gate = self._gate(context)
        else:
            gate = torch.full((x.size(0), 1), 0.5,
                              device=x.device, dtype=torch.float32)

        a_evade   = self.evade_out(F.relu(self.evade_hidden(h)))
        a_collect = self.collect_out(F.relu(self.collect_hidden(h)))
        v         = self.value_out(F.relu(self.value_hidden(h)))

        def _dueling(adv: torch.Tensor) -> torch.Tensor:
            return v + (adv - adv.mean(dim=1, keepdim=True))

        a_blend = gate * a_evade + (1.0 - gate) * a_collect
        return _dueling(a_evade), _dueling(a_collect), _dueling(a_blend), gate

    # ------------------------------------------------------------------
    # Warm-start from CNNDuelingQNetwork checkpoint
    # ------------------------------------------------------------------
    def load_from_cnn_dueling(
        self, path: str, map_location: str | None = None
    ) -> None:
        """
        Load weights from a DQNv2Agent checkpoint that used CNNDuelingQNetwork.

        Mapping:
          Shared trunk (conv1, conv2, fc_shared, value head) — copied directly.
          Old `adv_hidden/adv_out` → both evade AND collect heads (same init).
          Gate network — freshly initialized (random; learned from scratch).
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        # DQNv2Agent.save() wraps under "qnetwork_local"
        old = ckpt.get("qnetwork_local", ckpt)

        new = self.state_dict()
        direct = [
            "conv1.weight", "conv1.bias",
            "conv2.weight", "conv2.bias",
            "fc_shared.weight", "fc_shared.bias",
            "value_hidden.weight", "value_hidden.bias",
            "value_out.weight", "value_out.bias",
        ]
        for k in direct:
            if k in old:
                new[k] = old[k]

        # Old advantage head → both specialized heads
        adv_map = {
            "adv_hidden.weight": ["evade_hidden.weight", "collect_hidden.weight"],
            "adv_hidden.bias":   ["evade_hidden.bias",   "collect_hidden.bias"],
            "adv_out.weight":    ["evade_out.weight",    "collect_out.weight"],
            "adv_out.bias":      ["evade_out.bias",      "collect_out.bias"],
        }
        for old_key, new_keys in adv_map.items():
            if old_key in old:
                for nk in new_keys:
                    new[nk] = old[old_key]

        self.load_state_dict(new)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str | None = None) -> None:
        self.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True)
        )
