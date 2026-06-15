# Q6 — Kṛṣṇa vs Hunter: Deep RL Self-Play

A two-agent reinforcement learning project where **Kṛṣṇa** (a pellet-collecting agent) and **Hunter** (a chasing agent) both learn entirely through self-play. Built with PyTorch, Double DQN, Dueling networks, and Fictitious Self-Play (FSP).

> **For researchers and collaborators:** See [`versions/`](versions/README.md) for the full research log — each version's thesis, results, failure mode, and rationale for the next iteration. The live dashboard at `http://localhost:8080/dashboard/versions.html` shows the same information interactively.

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -q              # 147 tests

# Phase 1: Krishna learns vs scripted A* Hunter
python3 train_v2.py --episodes 6000 --device mps

# Phase 2: both agents learn via self-play
python3 train_phase2.py --episodes 6000 --device mps --name my_run

# Visualise results in browser
python3 dashboard/scan.py
python3 -m http.server 8080
# open http://localhost:8080/dashboard/
```

---


## The Game

25×25 gridworld with walls (~20% density). Krishna collects 4 pellets to win. Hunter catches Krishna 3 times to win. Episodes end on win, hunter-win, or 1000-step timeout.

```
Grid cell IDs:  WALL=0  PELLET=1  KRISHNA=2  HUNTER=3  EMPTY=6
Actions:        UP=0    DOWN=1    LEFT=2     RIGHT=3
```

---

## Phase 1 — Krishna vs Scripted Hunter

Krishna learns with a DQNv2 agent against a progressively harder scripted Hunter (random → greedy → A*).

**Network architecture** (`model/cnn_q_network.py`):
- Input: 6-channel 25×25 binary tensor (one channel per cell type)
- Conv: 32 filters (3×3) → 64 filters (3×3), ReLU
- FC: 256 hidden → Dueling heads: Value V(s) + Advantage A(s,a)
- Output: Q(s,a) for 4 actions

**Training:**
- Double DQN (decouple action selection from value estimation)
- Soft target update τ=0.001, Huber loss, Adam lr=1e-4
- ε-greedy: 1.0 → 0.05 at rate 0.9994/episode
- Replay buffer: 100k transitions, batch 64, update every 4 steps

**Result:** 85% win rate vs A* Hunter at 6000 episodes. (`tag: phase1-complete`)

---

## Phase 2 — Self-Play with Fictitious Self-Play (FSP)

Both Krishna and Hunter are live DQNv2 agents learning simultaneously.

### What is FSP?

Without FSP, two agents co-training will cycle: Krishna learns to beat the current Hunter, Hunter adapts, Krishna adapts back — policy cycling with no convergence. FSP breaks this by maintaining a historical snapshot pool.

Each episode is one of two modes:

- **Joint** (70%): Both agents step, observe, and learn from the same episode. Policies improve against each other's current strategy.
- **FSP** (30%): Krishna plays against a *frozen historical snapshot* of Hunter sampled from the pool. This forces Krishna to generalise across Hunter's full learning history, not just beat the current version.

Hunter snapshots are saved every 100 episodes (max 20, FIFO eviction → `pool/pool_index.json`).

### Reward design

**Krishna** needs both pellet skill and evasion — two competing pressures:

| Signal | Value | Why |
|---|---|---|
| Collect pellet | +50 | Primary objective |
| Win (4 pellets) | +100 | Terminal |
| Caught | −10 | Avoid Hunter |
| Lose (0 lives) | −50 | Terminal penalty |
| Step | −0.001 | Efficiency pressure |
| Proximity shaping | ±0.3 × Δd | Dense gradient toward nearest pellet |

**Hunter** has a single objective — chase:

| Signal | Value | Why |
|---|---|---|
| Catch | +30 | Primary objective |
| Win (3 catches) | +50 | Terminal |
| Krishna wins | −50 | Terminal penalty |
| Step | −0.001 | Efficiency pressure |
| Chase shaping | +0.1 × Δd | Dense gradient toward Krishna |

Wall hits carry **zero penalty** (`K_WALL=0`). A non-zero penalty (tried −5, then −1) created a pathological local attractor: the greedy policy learned to press into a corner wall for entire episodes, because the penalty was still lower than the expected cost of moving toward an aggressive Hunter. Zero penalty removes the attractor entirely while walls remain physically impassable.

### Epsilon decay calibration

With 6000 episodes and `ε_min=0.05`:

$$\text{decay} = \left(\frac{0.05}{1.0}\right)^{1/5000} \approx 0.9994$$

The old default of 0.9999 would leave ε≈0.55 at episode 6000 — the agent never exits exploration. At 0.9994, ε reaches the floor at episode ~5000, leaving 1000 episodes of near-pure exploitation.

### What actually emerged (v3 run, 6k episodes)

- **Both networks learned**: hunter_loss was nonzero from episode 10 onward (was flat 0.000 without approach shaping — Hunter had no dense gradient without it)
- **Krishna's first win**: episode 279 (vs episode 1049 in the previous run without fixes)
- **Hunter converged fast**: aggressive catch behaviour by ep ~500; catching in under 300 steps by ep 3000
- **Red Queen dynamics**: Krishna win 4% overall, 11% in the final 28 episodes as both ε values reached floor — late-game improvement confirms genuine learning, not luck
- **Remaining noise**: ~3.6% of episodes were catastrophic wall-hugging collapses (r_k ≤ −800); fixed in v4 with K_WALL=0

---

## Repository Layout

```
Q6/
├── agent/
│   ├── dqn_v2_agent.py       # DQNv2: CNN + Dueling + Double DQN
│   ├── frozen_agent.py       # Read-only checkpoint opponent for FSP
│   └── opponent_pool.py      # FSP snapshot pool (FIFO, max 20)
├── environment/
│   ├── hunter_gridworld.py   # Phase 1 env (scripted Hunter)
│   └── selfplay_env.py       # Phase 2 env (Hunter externally controlled)
├── model/
│   └── cnn_q_network.py      # CNN Dueling Q-network
├── utils/
│   ├── state_encoder.py      # 6-channel binary encoder (shared Phase 1+2)
│   ├── replay_recorder.py    # Per-step replay recording for dashboard
│   └── environment_wrapper.py
├── dashboard/
│   ├── scan.py               # Rebuilds index.json from training_runs/
│   ├── index.html            # Run list
│   ├── run.html              # Per-run metrics
│   └── replay.html           # Step-by-step replay viewer
├── tests/                    # 147 tests (pytest)
├── train_v2.py               # Phase 1 training
├── train_phase2.py           # Phase 2 self-play training
├── verify_phase2.py          # Checkpoint sanity checker
└── config.py                 # All hyperparameters centralised
```

---

## Key Hyperparameters (`config.py`)

| Parameter | Value | Note |
|---|---|---|
| `EPSILON_DECAY` | 0.9994 | Per-episode; reaches 0.05 floor ~ep 5000 |
| `EPSILON_MIN` | 0.05 | 5% random at convergence |
| `GAMMA` | 0.99 | Discount factor |
| `TAU` | 0.001 | Soft target update |
| `LEARNING_RATE` | 1e-4 | Adam |
| `BATCH_SIZE` | 64 | Replay sample size |
| `BUFFER_SIZE` | 100,000 | Per-agent replay capacity |
| `UPDATE_EVERY` | 4 | Steps between gradient updates |

---

## Lessons Learned

**1. Reward scale dominates early learning.**
A wall penalty of −5 completely drowned the pellet signal (+50). The agent learned nothing useful for hundreds of episodes. Reducing to −1 helped; setting to 0 eliminated the problem entirely. When your agent is doing something bizarre, check whether one reward term is an order of magnitude larger than the others.

**2. Dense shaping is not optional for sparse rewards.**
Without `K_APPROACH=0.3`, Krishna had zero gradient toward pellets for the first 200 episodes — every episode timed out and every update was noise. The shaping reward turns a sparse "collect pellet" signal into a dense continuous gradient that works from episode 1.

**3. Calibrate ε-decay to your episode budget.**
`decay = (ε_min / ε_start)^(1 / target_episode)`. If you miss this, the agent never exits exploration. This is one of the most common silent failures in DQN experiments.

**4. FSP prevents Nash cycling.**
Without the historical pool, two co-training agents cycle endlessly: A beats B, B adapts, A adapts, repeat. The pool makes each agent's policy robust to the full distribution of opponent strategies seen so far, not just the latest one.

**5. In asymmetric tasks, the simpler objective wins faster.**
Hunter (single goal: reduce distance) converged to aggressive chasing by episode 500. Krishna (two competing goals: collect pellets AND evade) was still learning at episode 6000. This is expected and correct — it is the Red Queen dynamic working as designed.

---

## Running Tests

```bash
python -m pytest tests/ -q                      # all 147 tests
python -m pytest tests/test_selfplay_env.py -v  # Phase 2 env
```

---

## Monitoring a Training Run

```bash
# Live log tail
tail -f training_runs/<run_dir>/logs/episode_stats.csv

# Quick summary of a completed run
python3 - <<'EOF'
import csv, statistics
rows = list(csv.DictReader(open("training_runs/<run_dir>/logs/episode_stats.csv")))
last = rows[-500:]
k = sum(1 for r in last if r["winner"]=="krishna")
print(f"Last 500: Krishna {k}/500 ({100*k/500:.1f}%)  avg_pellets {statistics.mean(float(r['pellets']) for r in last):.2f}")
EOF
```


TROUBLESHOOTING:
================

Problem: "Module not found" error
Solution: Make sure you activated venv: source venv/bin/activate

Problem: CUDA out of memory
Solution: Set device="cpu" in main.py (should auto-detect, but just in case)

Problem: Training seems stuck (scores not improving after 500 episodes)
Solution: 
- Check TRAINING_IMPROVEMENTS.txt for next debugging steps
- The -0.01 reward is the critical fix - make sure it's in place
- Watch intermediate metrics (pellets, hits, etc.) - they might be improving

Problem: Can't understand the output
Solution:
- Read QUICK_START.txt for output interpretation
- Watch a few episodes, note the metrics
- They'll start making sense!


NEXT STEPS AFTER FIRST SUCCESSFUL RUN:
========================================

If training converges and agent learns to win:

1. Analyze the learned behavior
   - Which actions does it prefer?
   - Where does it go on the map?
   - How does it evade enemies?

2. Test generalization
   - Does learned strategy work on different seed?
   - Can it handle layout variations?

3. Move to MVP 2 "Protean"
   - Test adaptation to changing environments
   - Implement continual learning techniques

The path to Kṛṣṇa's eventual mastery begins here!


GOOD LUCK!
==========

You now have a complete, working RL implementation.
The logging will show you exactly what's happening.
The code is heavily commented to teach concepts.
The metrics will prove the agent is learning.

This is real AI learning in action. Enjoy the journey!

Questions? Check the documentation files.
Something not working? Run validate.py to diagnose.
Ready to understand the code? Check main.py's comments.

Hari Om 🙏
