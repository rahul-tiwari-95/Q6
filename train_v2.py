"""
train_v2.py — Training driver for the CNN + Double DQN + Dueling stack.

Phase 1 baseline: trains a single Krishna agent against the existing
hand-coded enemies, using the new state encoder + network + agent.

Outputs into `training_runs/<timestamp>_v2/`:
    experiment.json          run metadata + final metrics
    checkpoints/             agent .pth files
    replays/episode_*.jsonl  recorded episodes (sampled)
    logs/episode_stats.csv   per-episode summary

Usage:
    python train_v2.py                    # full curriculum
    python train_v2.py --smoke            # 50 episodes, sanity check
    python train_v2.py --episodes 2000 --phase-allocation 500,500,500,500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

from config import (
    BATCH_SIZE, BUFFER_SIZE, EPSILON_DECAY, EPSILON_MIN, EPSILON_START,
    GAMMA, GRID_SIZE, LEARNING_RATE, PHASE_EPSILON_RESET, TAU, UPDATE_EVERY,
)
from agent.dqn_v2_agent import DQNv2Agent
from environment.hunter_gridworld import HunterGridworld
from utils.replay_recorder import ReplayRecorder


# ----------------------------- config -----------------------------

PHASE_DIFFICULTY = {1: 4, 2: 4, 3: 4, 4: 4}  # all phases use diff=4 (full game),
                                              # curriculum_phase varies hunter difficulty.


@dataclass
class TrainConfig:
    run_id: str
    run_dir: str
    phase_episodes: List[int]
    replay_every: int = 25      # record one episode every N
    checkpoint_every: int = 200
    log_every: int = 10
    max_steps_per_episode: int = 1000
    seed: int = 0


@dataclass
class EpisodeStats:
    episode: int
    phase: int
    curriculum_phase: int
    steps: int
    reward: float
    pellets: int
    caught: int
    won: bool
    epsilon: float
    loss: float | None
    mean_q: float | None


# ----------------------------- helpers -----------------------------

def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def _git_branch() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def _flatten_grid_to_2d(flat_state: np.ndarray, grid_size: int = GRID_SIZE) -> List[List[int]]:
    return flat_state.reshape(grid_size, grid_size).astype(int).tolist()


# ----------------------------- training -----------------------------

def train(cfg: TrainConfig, *, device: str | None = None) -> dict:
    """Run training and return the final experiment.json dict."""
    Path(cfg.run_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(cfg.run_dir, "logs").mkdir(parents=True, exist_ok=True)
    Path(cfg.run_dir, "replays").mkdir(parents=True, exist_ok=True)

    agent = DQNv2Agent(device=device)
    recorder = ReplayRecorder(run_dir=cfg.run_dir, enabled=True)

    started = datetime.now(timezone.utc)
    t0 = time.time()

    log_path = Path(cfg.run_dir, "logs", "episode_stats.csv")
    log_fh = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_fh, fieldnames=[
        "episode", "phase", "curriculum_phase", "steps", "reward",
        "pellets", "caught", "won", "epsilon", "loss", "mean_q",
    ])
    log_writer.writeheader()

    per_phase_results: list[dict] = []
    total_episodes = sum(cfg.phase_episodes)
    global_ep = 0

    rng = np.random.default_rng(cfg.seed)

    for phase_idx, n_ep in enumerate(cfg.phase_episodes):
        curriculum_phase = phase_idx + 1  # 1..4
        difficulty = PHASE_DIFFICULTY[curriculum_phase]

        # Reset epsilon per phase (matches existing convention).
        agent.reset_epsilon(PHASE_EPSILON_RESET.get(curriculum_phase, EPSILON_START))

        env = HunterGridworld(difficulty_level=difficulty,
                              curriculum_phase=curriculum_phase)

        phase_rewards: list[float] = []
        phase_wins = 0
        phase_pellets = 0
        phase_caught = 0

        print(f"\n=== Phase {curriculum_phase}: {n_ep} episodes "
              f"(difficulty={difficulty}, epsilon={agent.epsilon:.2f}) ===")

        for local_ep in range(1, n_ep + 1):
            global_ep += 1
            seed_for_episode = int(rng.integers(0, 2**31 - 1))
            state, _ = env.reset(seed=seed_for_episode)

            record_this = (global_ep % cfg.replay_every == 0) or (local_ep == n_ep)
            if record_this:
                recorder.start_episode(
                    episode_id=global_ep, phase=phase_idx + 1,
                    difficulty=difficulty, seed=seed_for_episode,
                    agents=["krishna"], grid_size=GRID_SIZE,
                )

            ep_reward = 0.0
            ep_steps = 0
            pellets_before = 0
            caught_before = 0

            for t in range(cfg.max_steps_per_episode):
                action = agent.act(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                agent.step(state, action, reward, next_state, done)

                ep_reward += float(reward)
                ep_steps += 1

                if record_this:
                    qs = agent.q_values(state)
                    recorder.record_step(
                        t=t, grid=_flatten_grid_to_2d(state),
                        actions={"krishna": int(action)},
                        rewards={"krishna": float(reward)},
                        q_values={"krishna": qs.tolist()},
                        lives=int(info.get("lives", env.krishna.lives if env.krishna else -1)),
                        pellets=int(info.get("pellets_collected", 0)),
                        done=bool(done or truncated),
                    )

                state = next_state
                if done or truncated:
                    break

            agent.decay_epsilon()

            pellets = int(info.get("pellets_collected", 0))
            won = pellets >= 4
            caught = int(info.get("times_caught", 0))
            phase_rewards.append(ep_reward)
            phase_wins += int(won)
            phase_pellets += pellets
            phase_caught += caught

            if record_this:
                recorder.end_episode(
                    outcome="win" if won else ("loss" if done else "timeout"),
                    total_reward={"krishna": ep_reward},
                    pellets_collected=pellets,
                )

            stats = EpisodeStats(
                episode=global_ep, phase=phase_idx + 1, curriculum_phase=curriculum_phase,
                steps=ep_steps, reward=ep_reward, pellets=pellets, caught=caught,
                won=won, epsilon=agent.epsilon,
                loss=agent.last_loss, mean_q=agent.last_mean_q,
            )
            log_writer.writerow(asdict(stats))
            log_fh.flush()

            if global_ep % cfg.log_every == 0:
                last100 = phase_rewards[-100:]
                print(f"  ep {global_ep:5d}/{total_episodes}  "
                      f"reward={ep_reward:8.2f}  pellets={pellets}  "
                      f"won={won}  eps={agent.epsilon:.3f}  "
                      f"avg100={np.mean(last100):.2f}")

            if global_ep % cfg.checkpoint_every == 0:
                ckpt_path = Path(cfg.run_dir, "checkpoints", f"agent_ep{global_ep:06d}.pth")
                agent.save(str(ckpt_path))

        per_phase_results.append({
            "phase": curriculum_phase,
            "episodes": n_ep,
            "win_rate": phase_wins / max(1, n_ep),
            "avg_reward": float(np.mean(phase_rewards)),
            "avg_pellets": phase_pellets / max(1, n_ep),
            "avg_caught": phase_caught / max(1, n_ep),
        })
        print(f"  -> phase {curriculum_phase} done: "
              f"win_rate={per_phase_results[-1]['win_rate']:.2%}, "
              f"avg_reward={per_phase_results[-1]['avg_reward']:.2f}")

    log_fh.close()
    final_ckpt = Path(cfg.run_dir, "checkpoints", "agent_final.pth")
    agent.save(str(final_ckpt))

    ended = datetime.now(timezone.utc)
    experiment = {
        "id": cfg.run_id,
        "name": f"Phase1 baseline — CNN+DDQN+Dueling",
        "phase_label": "phase1_foundation",
        "algo": {
            "name": "Double DQN + Dueling + CNN",
            "network": "CNNDuelingQNetwork(conv=[32,64], fc=256, head=128)",
            "state_encoding": "6-channel (6,25,25) binary",
            "loss": "smooth_l1",
        },
        "hyperparams": {
            "learning_rate": LEARNING_RATE, "gamma": GAMMA, "tau": TAU,
            "batch_size": BATCH_SIZE, "buffer_size": BUFFER_SIZE,
            "update_every": UPDATE_EVERY,
            "epsilon_start": EPSILON_START, "epsilon_min": EPSILON_MIN,
            "epsilon_decay": EPSILON_DECAY,
            "phase_epsilon_reset": PHASE_EPSILON_RESET,
            "phase_episodes": cfg.phase_episodes,
        },
        "env": {
            "grid_size": GRID_SIZE,
            "phases": [{"curriculum_phase": i + 1, "difficulty": PHASE_DIFFICULTY[i + 1]}
                       for i in range(len(cfg.phase_episodes))],
        },
        "training": {
            "total_episodes": total_episodes,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": time.time() - t0,
            "git_branch": _git_branch(),
            "git_commit": _git_commit(),
        },
        "results": {
            "per_phase": per_phase_results,
            "final_epsilon": agent.epsilon,
            "final_learning_step": agent.learning_step,
        },
        "artifacts": {
            "log_csv": str(log_path.relative_to(cfg.run_dir)),
            "final_checkpoint": str(final_ckpt.relative_to(cfg.run_dir)),
            "replays_dir": "replays",
        },
        "notes": "",
    }
    exp_path = Path(cfg.run_dir, "experiment.json")
    exp_path.write_text(json.dumps(experiment, indent=2))
    print(f"\nWrote {exp_path}")
    return experiment


# ----------------------------- CLI -----------------------------

def _parse_phase_alloc(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="50 ep total — sanity check")
    p.add_argument("--episodes", type=int, default=None, help="total episodes (overrides --phase-allocation)")
    p.add_argument("--phase-allocation", type=_parse_phase_alloc,
                   default=[3000, 3000, 3000, 9000],
                   help="comma-separated episodes per phase, e.g. 500,500,500,500")
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay-every", type=int, default=25)
    p.add_argument("--name", type=str, default="phase1_baseline")
    args = p.parse_args()

    if args.smoke:
        phase_alloc = [12, 12, 12, 14]  # 50 total
    elif args.episodes is not None:
        # Distribute roughly proportionally to default [1,1,1,3]
        n = args.episodes
        a = max(1, n // 6); b = max(1, n // 6); c = max(1, n // 6)
        d = n - (a + b + c)
        phase_alloc = [a, b, c, d]
    else:
        phase_alloc = args.phase_allocation

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.name}"
    run_dir = str(Path("training_runs") / run_id)

    cfg = TrainConfig(
        run_id=run_id, run_dir=run_dir,
        phase_episodes=phase_alloc,
        replay_every=args.replay_every,
        seed=args.seed,
    )
    print(f"Run dir: {run_dir}")
    print(f"Phase episodes: {phase_alloc} (total={sum(phase_alloc)})")
    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
