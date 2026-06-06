"""
Phase 2 warm-start sanity check.

Runs N episodes with Krishna loaded from a Phase 1 checkpoint and a random
Hunter, then prints a win/pellet summary.  Not a pytest test — run manually:

    python verify_phase2.py \
      --checkpoint training_runs/20260530_030934_phase1_baseline/checkpoints/agent_final.pth \
      --episodes 50 --device mps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.dqn_v2_agent import DQNv2Agent
from environment.selfplay_env import SelfPlayGridworld


def run(checkpoint: str, episodes: int, device: str) -> None:
    env = SelfPlayGridworld()
    krishna = DQNv2Agent(device=device)
    krishna.load(checkpoint)
    krishna.reset_epsilon(0.0)   # pure greedy — best-case probe

    rng = np.random.default_rng(42)
    wins = {"krishna": 0, "hunter": 0, "timeout": 0}
    pellet_list = []
    step_list = []

    for ep in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = trunc = False
        while not (done or trunc):
            a_k = krishna.act(state, training=False)
            a_h = int(rng.integers(0, 4))   # random Hunter
            state, _, done, trunc, info = env.step({"krishna": a_k, "hunter": a_h})
        winner = info.get("winner") or "timeout"
        wins[winner] = wins.get(winner, 0) + 1
        pellet_list.append(info["pellets_collected"])
        step_list.append(info["steps"])

    print(f"\n{'='*52}")
    print(f"Warm-start sanity check  ({episodes} ep, ε=0)")
    print(f"  checkpoint: {checkpoint}")
    print(f"  device: {device}")
    print(f"{'='*52}")
    print(f"  Krishna wins : {wins['krishna']:>4}  ({100*wins['krishna']/episodes:.1f}%)")
    print(f"  Hunter wins  : {wins['hunter']:>4}  ({100*wins['hunter']/episodes:.1f}%)")
    print(f"  Timeouts     : {wins['timeout']:>4}  ({100*wins['timeout']/episodes:.1f}%)")
    print(f"  Avg pellets  : {np.mean(pellet_list):.2f} / {env.TARGET_PELLETS}")
    print(f"  Avg steps    : {np.mean(step_list):.1f}")
    print(f"{'='*52}")
    # Pass/fail bar
    k_pct = wins["krishna"] / episodes
    if k_pct >= 0.40:
        print(f"  PASS — Krishna win rate {k_pct*100:.1f}% >= 40%")
    elif k_pct >= 0.20:
        print(f"  WARN — Krishna win rate {k_pct*100:.1f}% (20–40%); will improve with training)")
    else:
        print(f"  FAIL — Krishna win rate {k_pct*100:.1f}% < 20%; checkpoint may not transfer")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(args.checkpoint, args.episodes, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
