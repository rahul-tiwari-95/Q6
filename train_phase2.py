"""
Phase 2: Self-play training driver.

Trains Krishna and Hunter together as DQNv2 agents in `SelfPlayGridworld`.
Uses Fictitious Self-Play (FSP): with probability p_latest, Hunter is the
live learner; otherwise Krishna trains against a frozen historical Hunter
sampled from the opponent pool.

Hunter snapshots are added to the pool every `--snapshot-every` episodes.

Usage:
    python train_phase2.py --smoke
    python train_phase2.py --episodes 6000 --device mps --name phase2_selfplay
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Force unbuffered stdout so progress lines flush immediately under any shell.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from agent.dqn_v2_agent import DQNv2Agent
from agent.frozen_agent import FrozenAgent
from agent.opponent_pool import OpponentPool
from config import GRID_SIZE
from environment.selfplay_env import SelfPlayGridworld
from utils.replay_recorder import ReplayRecorder


REPO_ROOT = Path(__file__).resolve().parent


# ----------------------------- utilities -----------------------------

def _git(*args: str) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _new_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = REPO_ROOT / "training_runs" / f"{ts}_{name}"
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "replays").mkdir(parents=True, exist_ok=True)
    (d / "pool").mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------- training -----------------------------

def run_training(
    episodes: int,
    device: str | None,
    seed: int,
    name: str,
    snapshot_every: int,
    p_latest: float,
    replay_every: int,
    warm_start_krishna: str | None,
) -> Path:
    run_dir = _new_run_dir(name)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = SelfPlayGridworld(grid_size=GRID_SIZE)

    krishna = DQNv2Agent(device=device)
    hunter = DQNv2Agent(device=device)

    if warm_start_krishna:
        print(f"[warm-start] loading Krishna from {warm_start_krishna}")
        krishna.load(warm_start_krishna)
        # Reset epsilon so warm-started Krishna still explores against a learning hunter
        krishna.reset_epsilon(0.5)

    pool = OpponentPool(run_dir / "pool", max_size=20)
    # Seed the pool with the hunter's initial random network so FSP episodes
    # can run from episode 1.
    pool.add_snapshot(hunter, metadata={"episode": 0, "kind": "init"})

    recorder = ReplayRecorder(str(run_dir), enabled=True)

    log_path = run_dir / "logs" / "episode_stats.csv"
    log_file = log_path.open("w", newline="")
    log = csv.writer(log_file)
    log.writerow([
        "episode", "mode", "steps", "krishna_reward", "hunter_reward",
        "pellets", "caught", "winner", "krishna_eps", "hunter_eps",
        "krishna_loss", "hunter_loss", "pool_size",
    ])

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    last_print_t = t0

    # Per-mode counters for end-of-run summary
    wins = {"krishna": 0, "hunter": 0, "timeout": 0}
    mode_counts = {"joint": 0, "fsp": 0}

    print(f"[start] {episodes} episodes  device={krishna.device}  "
          f"p_latest={p_latest}  snapshot_every={snapshot_every}  run_dir={run_dir}",
          flush=True)

    rolling = []  # last-100 krishna rewards

    for ep in range(1, episodes + 1):
        mode = "joint" if rng.random() < p_latest else "fsp"
        mode_counts[mode] += 1

        # In FSP mode, Hunter is a frozen snapshot and does not learn
        if mode == "fsp":
            snap = pool.sample(rng, p_latest=0.3)  # bias toward older when in FSP mode
            hunter_actor: object = FrozenAgent.load(snap, device=krishna.device)
            hunter_learns = False
        else:
            hunter_actor = hunter
            hunter_learns = True

        ep_seed = int(rng.integers(0, 2**31 - 1))
        state, info = env.reset(seed=ep_seed)

        # Replay recording (every Nth)
        record_this = (ep % max(1, replay_every) == 0) or (ep == episodes)
        if record_this:
            recorder.start_episode(
                episode_id=ep,
                phase=0 if mode == "joint" else 1,
                difficulty=0,
                seed=ep_seed,
                grid_size=env.grid_size,
                agents=["krishna", "hunter"],
            )

        ep_r_k = 0.0
        ep_r_h = 0.0
        steps = 0
        done = trunc = False

        while not (done or trunc):
            a_k = krishna.act(state, training=True)
            a_h = hunter_actor.act(state, training=True)

            next_state, rewards, done, trunc, info = env.step({"krishna": a_k, "hunter": a_h})

            # Learn
            krishna.step(state, a_k, rewards["krishna"], next_state, done or trunc)
            if hunter_learns:
                hunter.step(state, a_h, rewards["hunter"], next_state, done or trunc)

            if record_this:
                recorder.record_step(
                    t=steps,
                    grid=env.grid.tolist(),
                    actions={"krishna": int(a_k), "hunter": int(a_h)},
                    rewards={"krishna": float(rewards["krishna"]),
                             "hunter": float(rewards["hunter"])},
                    q_values={
                        "krishna": krishna.q_values(state).tolist(),
                        "hunter": hunter_actor.q_values(state).tolist(),
                    },
                    lives=info["lives"],
                    pellets=info["pellets_collected"],
                    done=bool(done or trunc),
                )

            state = next_state
            ep_r_k += rewards["krishna"]
            ep_r_h += rewards["hunter"]
            steps += 1

        winner = info.get("winner") or "timeout"
        wins[winner] = wins.get(winner, 0) + 1

        if record_this:
            recorder.end_episode(
                outcome=winner,
                total_reward={"krishna": float(ep_r_k), "hunter": float(ep_r_h)},
                pellets_collected=info["pellets_collected"],
            )

        # Decay epsilons (both)
        krishna.decay_epsilon()
        if hunter_learns:
            hunter.decay_epsilon()

        # FSP snapshot
        if ep % snapshot_every == 0:
            pool.add_snapshot(hunter, metadata={"episode": ep})

        rolling.append(ep_r_k)
        if len(rolling) > 100:
            rolling.pop(0)
        avg100 = float(np.mean(rolling))

        log.writerow([
            ep, mode, steps,
            f"{ep_r_k:.3f}", f"{ep_r_h:.3f}",
            info["pellets_collected"], info["times_caught"], winner,
            f"{krishna.epsilon:.4f}", f"{hunter.epsilon:.4f}",
            f"{krishna.last_loss or 0.0:.5f}",
            f"{hunter.last_loss or 0.0:.5f}",
            len(pool),
        ])

        if ep % 10 == 0 or ep == episodes:
            log_file.flush()
            now = time.time()
            ep_per_sec = 10.0 / max(1e-6, (now - last_print_t)) if ep > 0 else 0.0
            last_print_t = now
            print(f"  ep {ep:>5}/{episodes}  mode={mode}  steps={steps:>4}  "
                  f"r_k={ep_r_k:>8.2f}  r_h={ep_r_h:>8.2f}  pellets={info['pellets_collected']}  "
                  f"winner={winner:<8}  pool={len(pool):>2}  avg100={avg100:.2f}  "
                  f"({ep_per_sec:.2f} ep/s)", flush=True)
    # ----- finalize -----
    log_file.close()
    recorder.close()

    ckpt_k = run_dir / "checkpoints" / "krishna_final.pth"
    ckpt_h = run_dir / "checkpoints" / "hunter_final.pth"
    krishna.save(str(ckpt_k))
    hunter.save(str(ckpt_h))

    ended_at = datetime.now(timezone.utc).isoformat()
    duration = time.time() - t0

    experiment = {
        "id": run_dir.name,
        "name": f"Phase2 selfplay — {name}",
        "phase_label": "phase2_selfplay",
        "algo": {
            "name": "Double DQN + Dueling + CNN (self-play, FSP)",
            "network": "CNNDuelingQNetwork(conv=[32,64], fc=256, head=128) x2",
            "state_encoding": "6-channel (6,25,25) binary",
            "loss": "smooth_l1",
            "fsp": {"p_latest_joint": p_latest, "snapshot_every": snapshot_every,
                    "pool_max_size": 20},
        },
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "tau": 0.001,
            "batch_size": 64, "buffer_size": 100_000, "update_every": 4,
            "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.9999,
        },
        "env": {
            "type": "SelfPlayGridworld",
            "grid_size": env.grid_size,
            "pellets": env.TARGET_PELLETS,
            "lives": env.KRISHNA_LIVES,
            "max_steps": env.MAX_STEPS,
        },
        "training": {
            "total_episodes": episodes,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration,
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit": _git("rev-parse", "HEAD"),
        },
        "results": {
            "wins": wins,
            "mode_counts": mode_counts,
            "krishna_win_rate": wins["krishna"] / max(1, episodes),
            "hunter_win_rate": wins["hunter"] / max(1, episodes),
            "timeout_rate": wins["timeout"] / max(1, episodes),
            "final_krishna_epsilon": krishna.epsilon,
            "final_hunter_epsilon": hunter.epsilon,
            "final_pool_size": len(pool),
        },
        "artifacts": {
            "log_csv": "logs/episode_stats.csv",
            "krishna_checkpoint": "checkpoints/krishna_final.pth",
            "hunter_checkpoint": "checkpoints/hunter_final.pth",
            "replays_dir": "replays",
            "pool_dir": "pool",
        },
        "notes": "",
    }
    (run_dir / "experiment.json").write_text(json.dumps(experiment, indent=2))
    print(f"\nWrote {run_dir / 'experiment.json'}")
    print(f"  krishna_win={wins['krishna']}  hunter_win={wins['hunter']}  "
          f"timeout={wins['timeout']}  duration={duration:.1f}s")
    return run_dir


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--smoke", action="store_true",
                   help="Run 40 episodes for pipeline validation.")
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="phase2_selfplay")
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--p-latest", type=float, default=0.7,
                   help="P(joint training episode); rest are FSP vs frozen.")
    p.add_argument("--replay-every", type=int, default=25)
    p.add_argument("--warm-start-krishna", default=None,
                   help="Path to Phase 1 Krishna checkpoint (qnetwork_local/target).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    episodes = 40 if args.smoke else args.episodes
    if args.smoke:
        args.snapshot_every = max(10, args.snapshot_every // 10)
        args.replay_every = 10
    run_training(
        episodes=episodes,
        device=args.device,
        seed=args.seed,
        name=args.name,
        snapshot_every=args.snapshot_every,
        p_latest=args.p_latest,
        replay_every=args.replay_every,
        warm_start_krishna=args.warm_start_krishna,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
