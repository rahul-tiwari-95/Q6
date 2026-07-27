"""
Integration check for train_v8.py's done-flag bookkeeping — the highest-risk
area flagged in the v8 PPO review (see Q6.md section 4 for why this project
takes "verify before you trust a multi-thousand-episode run" seriously).

This does not re-verify PPOAgent's GAE math (see tests/test_ppo_agent.py for
that) — it verifies that train_v8.py's training LOOP feeds `store()` the
right done flags across real episode boundaries, including the FSP-gap case
for Hunter (Hunter only stores while `hunter_learns`, so its call sequence
has gaps corresponding to skipped FSP-mode episodes; a staleness bug there
wouldn't show up in any single-agent PPOAgent-level test).
"""

from __future__ import annotations

import numpy as np

import train_v8
from agent.ppo_agent import PPOAgent
from environment.selfplay_env import SelfPlayGridworld


def test_done_flag_grouping_invariant_across_short_run(tmp_path, monkeypatch):
    """
    Every contiguous run of PPOAgent.store() calls on the SAME agent instance
    must begin with done=True (the state was freshly reset) and continue with
    done=False until that episode ends. This holds for Krishna (stores every
    step, every episode) and for Hunter (stores only during joint-mode
    episodes) alike — Hunter's h_prev_done must not go stale across one or
    more skipped FSP episodes.
    """
    # Keep episodes short so this runs in seconds, not minutes — we only care
    # about the done-flag bookkeeping, not real gameplay dynamics.
    monkeypatch.setattr(SelfPlayGridworld, "MAX_STEPS", 20)
    # Route training_runs/ output into tmp_path instead of the real repo.
    monkeypatch.setattr(train_v8, "REPO_ROOT", tmp_path)

    calls: dict[int, list[bool]] = {}
    original_store = PPOAgent.store

    def _tracking_store(self, state, action, logprob, reward, done, value):
        calls.setdefault(id(self), []).append(bool(done))
        return original_store(self, state, action, logprob, reward, done, value)

    monkeypatch.setattr(PPOAgent, "store", _tracking_store)

    # p_latest=0.5 over 24 episodes makes "every episode landed in the same
    # mode" astronomically unlikely (~2 * 0.5**24), so both Krishna's
    # (always-on) and Hunter's (gappy) call sequences get exercised for real.
    train_v8.run_training(
        episodes=24, device="cpu", seed=7, name="done_flag_check",
        snapshot_every=4, p_latest=0.5, replay_every=100, rollout_len=64,
    )

    assert len(calls) == 2, f"expected exactly 2 PPOAgent instances to call store(), got {len(calls)}"

    call_lengths = sorted(len(d) for d in calls.values())
    assert call_lengths[0] < call_lengths[1], (
        "expected Hunter (fewer calls, gapped by FSP episodes) and Krishna "
        f"(one call per env step, every episode) to have different total "
        f"call counts, got {call_lengths} — did every episode land in the "
        "same mode? (should be astronomically unlikely with this seed/count)"
    )

    for agent_id, dones in calls.items():
        assert dones[0] is True, f"agent {agent_id}: first-ever stored transition must have done=True"

        groups: list[list[bool]] = []
        current: list[bool] = []
        for d in dones:
            if d and current:
                groups.append(current)
                current = [d]
            else:
                current.append(d)
        if current:
            groups.append(current)

        assert len(groups) > 1, f"agent {agent_id}: expected multiple episode groups, got {len(groups)}"
        for g in groups:
            assert g[0] is True, f"agent {agent_id}: episode group did not start with done=True: {g[:5]}"
            assert all(x is False for x in g[1:]), (
                f"agent {agent_id}: found done=True mid-episode (should only occur at a group's "
                f"first element): {g}"
            )
