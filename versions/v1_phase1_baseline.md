# v1 — Phase 1: Single-Agent Baseline

**Date:** December 2025  
**Episodes:** ~6,000  
**Status:** Complete

---

## Thesis

A vanilla Double DQN with dueling heads and CNN state encoding can learn a viable
evasion + pellet-collection policy against a **scripted (non-learning) Hunter** in
a 25×25 grid world with sparse rewards.

## Architecture

| Component | Detail |
|---|---|
| Network | CNNDuelingQNetwork: conv(32→64) → FC(256) → advantage + value |
| State | 6-channel (6,25,25) binary grid |
| Hunter | Scripted greedy (deterministic BFS pathfinding) |
| Exploration | ε-greedy 1.0 → 0.05 over 6000 eps |
| Loss | Smooth L1 |
| Optimizer | Adam, lr=1e-4 |

## Key Files

- `agent/dq_agent.py` — DQNAgent (Double DQN + Dueling)
- `model/q_network.py` — CNNDuelingQNetwork
- `environment/hunter_gridworld.py` — single-agent env
- `environment/entities/krishna.py`, `hunter.py`

## Results

| Metric | Value |
|---|---|
| pellets/episode (avg) | ~0.8 |
| win rate | ~12% |
| avg100 reward | ~+15 |
| times caught / 1000 eps | ~40 |

## What Worked

- DQN training converged — loss declined, avg100 trended positive
- Krishna learned basic evasion patterns
- CNN state encoding was sufficient for single-agent task

## Failure Mode

Against a scripted Hunter, Krishna over-fit to the **deterministic pathfinding
pattern**. The scripted Hunter always moves along the same shortest-path, making
it trivially avoidable once memorized. This performance does not generalize:
replacing the scripted Hunter with any learned policy (even a random one) causes
Krishna's reward to collapse.

The agent did not learn a general "collect while evading" policy — it learned
"avoid specific grid positions the scripted Hunter visits."

## Decision for v2

**Need a co-evolving opponent.** The Hunter must also learn and adapt, so Krishna
is forced to develop general strategies rather than memorizing fixed patterns.

→ Introduce **Fictitious Self-Play (FSP)**: both agents train simultaneously,
Hunter snapshots are pooled, and Krishna faces diverse opponent behaviors.
