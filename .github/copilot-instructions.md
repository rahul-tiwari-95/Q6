# Copilot Instructions — Q6 (Kṛṣṇa vs Hunter)

## Project Overview

Two-agent deep RL self-play in a 25×25 gridworld. **Krishna** collects 4 pellets to win; **Hunter** catches Krishna 3 times to win. Built with PyTorch, Double DQN, Dueling networks, and Fictitious Self-Play (FSP).

---

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests
python -m pytest tests/ -q                          # full suite (147 tests)
python -m pytest tests/test_selfplay_env.py -v      # single file
python -m pytest tests/test_cher.py::TestCHER::test_basic_relabeling -v  # single test

# Training
python3 train_v2.py --episodes 6000 --device mps          # Phase 1: Krishna vs scripted Hunter
python3 train_phase2.py --episodes 6000 --device mps --name my_run  # Phase 2: self-play
python3 train_phase3.py --episodes 12000 --device mps --name phase3_v1_gop_cher \
  --warm-start training_runs/<run>/checkpoints/krishna_best_ep5300.pth \
  --epsilon-start 0.15                                      # Phase 3: GOP + CHER

# Dashboard
python3 dashboard/scan.py && python3 -m http.server 8080
# open http://localhost:8080/dashboard/

# Monitoring a live run
tail -f training_runs/<run_dir>/logs/episode_stats.csv
```

---

## Architecture

Training has three phases, each building on the last:

- **Phase 1** (`train_v2.py`): Krishna trains against a scripted Hunter (random → greedy → A*) using `HunterGridworld`.
- **Phase 2** (`train_phase2.py`): Both agents are live DQN learners. FSP (Fictitious Self-Play) prevents policy cycling by having Krishna occasionally play against frozen historical Hunter snapshots from `OpponentPool` (single-tier FIFO, max 20).
- **Phase 3** (`train_phase3.py`): Adds GOP (Gated Option Policies) + CHER (Counterfactual HER) + `HierarchicalOpponentPool` (2-tier: 5 permanent easy + 15 FIFO hard) to fix collection paralysis — Krishna's failure mode of evading forever without collecting pellets.

### Key Data Flow

```
SelfPlayGridworld → Training Loop → GatedDQNAgent (Krishna) + DQNv2Agent (Hunter)
                                  ↓ after each episode
                              CounterfactualHER.relabel() → inject CF transitions → ContextReplayBuffer
                                  ↓ every 100 episodes
                              HierarchicalOpponentPool.add_snapshot()
```

### State Representation

The flat `(625,)` int32 grid (values 0–6) is encoded into a `(6, 25, 25)` binary tensor by `utils/state_encoder.py`. One channel per entity type; EMPTY (6) is implicit (all-zeros). This encoding is **shared across all phases**.

Grid cell constants (from `config.py`): `WALL=0 PELLET=1 KRISHNA=2 HUNTER=3 GREEDY=4 PATROL=5 EMPTY=6`  
Actions: `UP=0 DOWN=1 LEFT=2 RIGHT=3`

### Gated Option Network (Phase 3)

`model/gated_option_network.py` — a single CNN backbone with two specialized heads (evade / collect) and a learned gate MLP that blends them:

- `gate=1.0` → pure evasion (Hunter nearby, low lives)
- `gate=0.0` → pure collection (Hunter far, pellets remain)

Gate is trained end-to-end through Q-learning loss; no separate supervised signal. Monitor `mean_gate` column in episode_stats.csv to detect gate collapse.

### Counterfactual HER (Phase 3)

`utils/cher.py` — post-episode relabeling. For steps where `hunter_dist > 6` AND `pellet_dist ≤ 4` AND Krishna didn't move toward a pellet: inject a counterfactual transition with the optimal collection action and `reward + collect_bonus(30.0)`. Same `state`/`next_state` as the real transition — only `action` and `reward` differ.

---

## Key Conventions

### All hyperparameters live in `config.py`

Never hardcode grid constants, reward values, or training hyperparameters. Import from `config`. Phase 3 overrides (e.g. `epsilon_start=0.15`, `cher_bonus=30.0`) are set in `train_phase3.py` via CLI args.

### Replay buffer stores raw uint8 states

`FlatReplayBuffer` (Phase 2) and `ContextReplayBuffer` (Phase 3) store flat `uint8` arrays, not encoded tensors. Channel encoding (`encode_batch`) runs at sample time on the device. This saves ~6x memory.

### Epsilon decay must be calibrated to episode budget

`decay = (ε_min / ε_start) ^ (1 / target_episode)`. The default `EPSILON_DECAY=0.9994` reaches `ε_min=0.05` at ~ep 5000 over a 6000-episode run. Using the wrong decay (e.g. 0.9999) leaves ε≈0.55 at ep 6000 — agent never exploits.

### Wall penalty must stay at zero (`K_WALL=0` in Phase 2+)

A non-zero wall penalty (even −1) creates a pathological local attractor where the agent hugs a corner wall for entire episodes. `REWARD_WALL_HIT` in `config.py` is kept for Phase 1 only; Phase 2+ training scripts explicitly pass `k_wall=0`.

### Training outputs go to `training_runs/`

Each run creates `training_runs/<timestamp>_<name>/` with subdirs `logs/`, `checkpoints/`, `plots/`, and optionally `pool/`. The dashboard's `scan.py` rebuilds `index.json` by scanning this directory.

### `FrozenAgent` is the FSP opponent wrapper

`agent/frozen_agent.py` wraps a saved checkpoint as a read-only opponent. Both `OpponentPool` (Phase 2) and `HierarchicalOpponentPool` (Phase 3) return paths; the training loop loads them via `FrozenAgent.load(path)`.

### Phase 3 warm-starts from Phase 2 checkpoint

Pass the Phase 2 best checkpoint via `--warm-start` to `train_phase3.py`. Set `--epsilon-start 0.15` (near floor) since the network already has a learned policy.
