"""
Independent PPO agent for adversarial self-play (v8).

Not a drop-in replacement for DQNv2Agent's per-step `.step()` API: PPO is
on-policy, so it collects a fixed-length rollout, computes GAE advantages
over the whole rollout, and updates in shuffled minibatched epochs — there
is no persistent replay buffer and no "learn every N steps."

Hyperparameters default to the CleanRL CNN-PPO reference (ppo_atari.py) as
closely as this single, unvectorized environment allows, since v8's whole
point is testing whether Q6's risk-dominant policy-bias collapse is a
DQN-specific artifact — an idiosyncratic PPO config would confound that
comparison. See Q6.md section 3.3 and 3.4.

Done-flag convention (read this before touching the rollout loop)
-------------------------------------------------------------------
Following CleanRL's convention exactly, to avoid an off-by-one GAE bug:
the `done` passed to `store()` for a transition describes whether the
*state being acted from* is itself the fresh result of an episode reset —
i.e. it is the termination flag carried over from the *previous* stored
transition, not whether the current action ends the episode. Concretely,
in the training loop:

    k_prev_done = True                       # training start: trivially "post-reset"
    ...
    action, logprob, value = agent.act(state)
    next_state, reward, done, trunc, info = env.step(...)
    ep_done = done or trunc
    agent.store(state, action, logprob, reward, k_prev_done, value)
    k_prev_done = ep_done                    # becomes the flag for the NEXT stored transition
    ...
    if agent.ready_to_update():
        next_value = agent.get_value(state)  # state = the NEXT state to act from
        agent.update(next_value=next_value, next_done=k_prev_done)

This lets `update()`'s GAE backward pass zero the bootstrap correctly across
every episode boundary inside the buffer, including boundaries that don't
line up with the update call itself (rollouts routinely span partial and
multiple episodes).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import ACTION_SIZE, GRID_SIZE
from model.actor_critic_network import CNNActorCritic
from utils.state_encoder import NUM_CHANNELS, encode_batch, encode_state


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-length on-policy buffer for a single (unvectorized) environment."""

    def __init__(self, rollout_len: int) -> None:
        self.rollout_len = int(rollout_len)
        self.reset()

    def reset(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.uint8))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.states)

    def full(self) -> bool:
        return len(self.states) >= self.rollout_len


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        action_size: int = ACTION_SIZE,
        grid_size: int = GRID_SIZE,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,     # CleanRL's CNN/Atari default (0.2 for MLP-only envs)
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        rollout_len: int = 2048,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.action_size = action_size
        self.grid_size = grid_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.network = CNNActorCritic(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        self.buffer = RolloutBuffer(rollout_len)

        # Diagnostics (mirrors the DQN agents' last_loss/last_mean_q convention
        # so dashboard/logging code can treat both agent families uniformly).
        self.update_step: int = 0
        self.last_loss: Optional[float] = None
        self.last_policy_loss: Optional[float] = None
        self.last_value_loss: Optional[float] = None
        self.last_entropy: Optional[float] = None
        self.last_approx_kl: Optional[float] = None
        self.last_clipfrac: Optional[float] = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Sample (action, log_prob, value) from the current stochastic policy.

        `training` is accepted for interface parity with DQNv2Agent /
        GatedDQNAgent (which use it to gate epsilon-greedy exploration) but
        is otherwise unused here — PPO always samples during rollout
        collection; there is no separate greedy-eval action path on this
        class (use `act_greedy` for that).
        """
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            action, logprob, _entropy, value = self.network.get_action_and_value(x)
        return int(action.item()), float(logprob.item()), float(value.item())

    def act_greedy(self, state: np.ndarray) -> int:
        """Argmax action (no sampling noise) — used for evaluation."""
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            logits, _value = self.network(x)
        return int(logits.argmax(dim=1).item())

    def get_value(self, state: np.ndarray) -> float:
        """Value estimate only — used to bootstrap GAE at a rollout boundary."""
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            _logits, value = self.network(x)
        return float(value.item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Policy logits, NOT Q-values — exposed under this name only for
        interface parity with DQNv2Agent.q_values()/FrozenAgent.q_values()
        so the replay recorder and dashboard can call it uniformly across
        agent families. Callers displaying this should label the axis
        "policy logit", not "Q-value".
        """
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            logits, _value = self.network(x)
        return logits.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """See the module docstring for the exact meaning of `done` here."""
        self.buffer.add(state, action, logprob, reward, done, value)

    def ready_to_update(self) -> bool:
        return self.buffer.full()

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, next_value: float, next_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                nextnonterminal = 1.0 - float(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, next_value: float, next_done: bool) -> Dict[str, float]:
        """
        Run `update_epochs` of shuffled-minibatch clipped-surrogate PPO over
        the current buffer, then reset it. `next_value`/`next_done` bootstrap
        GAE past the last stored transition — see the module docstring for
        how the caller should compute them.
        """
        n = len(self.buffer)
        if n == 0:
            raise RuntimeError("update() called on an empty buffer")

        advantages, returns = self._compute_gae(next_value, next_done)

        states_np = np.stack(self.buffer.states).astype(np.int32)
        b_states = torch.from_numpy(encode_batch(states_np, self.grid_size)).to(self.device)
        b_actions = torch.as_tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        b_logprobs = torch.as_tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        b_values = torch.as_tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        b_advantages = torch.from_numpy(advantages).to(self.device)
        b_returns = torch.from_numpy(returns).to(self.device)

        minibatch_size = max(1, n // self.num_minibatches)
        b_inds = np.arange(n)
        clipfracs: List[float] = []

        pg_loss = v_loss = entropy_loss = loss = None
        approx_kl = torch.tensor(0.0)

        for _epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, n, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                if len(mb_inds) == 0:
                    continue

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    b_states[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.update_step += 1
        self.last_loss = float(loss.item())
        self.last_policy_loss = float(pg_loss.item())
        self.last_value_loss = float(v_loss.item())
        self.last_entropy = float(entropy_loss.item())
        self.last_approx_kl = float(approx_kl.item())
        self.last_clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0

        self.buffer.reset()
        return {
            "loss": self.last_loss,
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "approx_kl": self.last_approx_kl,
            "clipfrac": self.last_clipfrac,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves the network state_dict only (no optimizer/buffer state) —
        matches what FrozenPPOAgent.load() and the FSP opponent pool expect."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt)
