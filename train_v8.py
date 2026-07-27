"""
v8: Independent PPO self-play training driver.

Reproduces Phase 2's exact experimental design (`SelfPlayGridworld` with its
current, unmodified reward constants; `OpponentPool` FIFO snapshots; the same
joint/FSP episode split) with exactly ONE variable changed: the learning
algorithm, DQNv2Agent -> PPOAgent. This isolates whether the risk-dominant
policy-bias collapse documented in Q6.md (Krishna converging to "survive,
don't collect" against a hardened Hunter) is a DQN-specific artifact — its
replay buffer keeps sampling transitions generated against stale, long-past
opponent policies — or a structural property of the game itself, independent
of the learner. See Q6.md sections 3.3 and 3.4 for the full reasoning.

Do NOT add reward shaping, curriculum, or architecture changes here — that
would confound the exact comparison this run exists to make. If you want to
test a fix, that belongs on a different branch (see Q6.md section 9).

Usage:
    python train_v8.py --smoke
    python train_v8.py --episodes 6000 --device mps --name v8_ippo_selfplay
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

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from agent.frozen_ppo_agent import FrozenPPOAgent
from agent.opponent_pool import OpponentPool
from agent.ppo_agent import PPOAgent
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
    rollout_len: int,
) -> Path:
    run_dir = _new_run_dir(name)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # PPOAgent.update() shuffles minibatches via the legacy global numpy RNG
    # (np.random.shuffle), which the local `rng` Generator above does not
    # cover — seed it too so --seed is fully reproducible, not just episode
    # ordering. (Flagged in the v8 PPO review; see Q6.md section 4 for why
    # reproducibility gaps get taken seriously in this repo.)
    np.random.seed(seed)

    env = SelfPlayGridworld(grid_size=GRID_SIZE)

    krishna = PPOAgent(device=device, rollout_len=rollout_len)
    hunter = PPOAgent(device=device, rollout_len=rollout_len)

    pool = OpponentPool(run_dir / "pool", max_size=20)
    # Seed the pool with Hunter's initial random network so FSP episodes can
    # run from episode 1 (matches train_phase2.py).
    pool.add_snapshot(hunter, metadata={"episode": 0, "kind": "init"})

    recorder = ReplayRecorder(str(run_dir), enabled=True)

    log_path = run_dir / "logs" / "episode_stats.csv"
    log_file = log_path.open("w", newline="")
    log = csv.writer(log_file)
    log.writerow([
        "episode", "mode", "steps", "krishna_reward", "hunter_reward",
        "pellets", "caught", "winner",
        "krishna_updates", "hunter_updates",
        "krishna_loss", "hunter_loss",
        "krishna_entropy", "hunter_entropy",
        "krishna_approx_kl", "hunter_approx_kl",
        "krishna_clipfrac", "hunter_clipfrac",
        "pool_size", "avg100",
    ])

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    last_print_t = t0

    wins = {"krishna": 0, "hunter": 0, "timeout": 0}
    mode_counts = {"joint": 0, "fsp": 0}
    rolling: list[float] = []

    print(f"[start] {episodes} episodes  device={krishna.device}  algo=IPPO  "
          f"p_latest={p_latest}  snapshot_every={snapshot_every}  "
          f"rollout_len={rollout_len}  run_dir={run_dir}", flush=True)

    # ---- episode/mode setup (shared by the initial episode and every
    # subsequent one picked at an episode boundary below) ----
    def _pick_mode() -> tuple[str, object, bool]:
        m = "joint" if rng.random() < p_latest else "fsp"
        mode_counts[m] += 1
        if m == "fsp":
            snap = pool.sample(rng, p_latest=0.3)  # bias toward older when in FSP mode
            actor = FrozenPPOAgent.load(snap, device=krishna.device)
            return m, actor, False
        return m, hunter, True

    mode, hunter_actor, hunter_learns = _pick_mode()
    ep_seed = int(rng.integers(0, 2**31 - 1))
    state, info = env.reset(seed=ep_seed)

    ep = 0
    ep_r_k = 0.0
    ep_r_h = 0.0
    ep_steps = 0
    # See agent/ppo_agent.py module docstring for the exact meaning of these
    # "done" flags — they describe whether the state about to be acted on is
    # itself the fresh result of an episode reset, not whether this step ends
    # the episode. Always True at the start of a new episode.
    k_prev_done = True
    h_prev_done = True

    record_this = True  # always record episode 1
    if record_this:
        recorder.start_episode(
            episode_id=1, phase=0 if mode == "joint" else 1, difficulty=0,
            seed=ep_seed, grid_size=env.grid_size, agents=["krishna", "hunter"],
        )

    k_updates_this_ep = 0
    h_updates_this_ep = 0

    while ep < episodes:
        a_k, logp_k, v_k = krishna.act(state, training=True)
        if hunter_learns:
            a_h, logp_h, v_h = hunter.act(state, training=True)
        else:
            a_h = hunter_actor.act(state, training=True)

        next_state, rewards, done, trunc, info = env.step({"krishna": a_k, "hunter": a_h})
        ep_done = bool(done or trunc)

        krishna.store(state, a_k, logp_k, rewards["krishna"], k_prev_done, v_k)
        k_prev_done = ep_done

        if hunter_learns:
            hunter.store(state, a_h, logp_h, rewards["hunter"], h_prev_done, v_h)
            h_prev_done = ep_done

        if record_this:
            recorder.record_step(
                t=ep_steps,
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
                done=ep_done,
            )

        ep_r_k += rewards["krishna"]
        ep_r_h += rewards["hunter"]
        ep_steps += 1
        state = next_state

        # ---- PPO updates trigger whenever a buffer fills, independent of
        # episode boundaries — rollouts routinely span partial episodes. ----
        if krishna.ready_to_update():
            nv = krishna.get_value(state)
            krishna.update(next_value=nv, next_done=k_prev_done)
            k_updates_this_ep += 1
        if hunter_learns and hunter.ready_to_update():
            nv = hunter.get_value(state)
            hunter.update(next_value=nv, next_done=h_prev_done)
            h_updates_this_ep += 1

        if ep_done:
            ep += 1
            winner = info.get("winner") or "timeout"
            wins[winner] = wins.get(winner, 0) + 1

            if record_this:
                recorder.end_episode(
                    outcome=winner,
                    total_reward={"krishna": float(ep_r_k), "hunter": float(ep_r_h)},
                    pellets_collected=info["pellets_collected"],
                )

            if ep % snapshot_every == 0:
                pool.add_snapshot(hunter, metadata={"episode": ep})

            rolling.append(ep_r_k)
            if len(rolling) > 100:
                rolling.pop(0)
            avg100 = float(np.mean(rolling))

            log.writerow([
                ep, mode, ep_steps,
                f"{ep_r_k:.3f}", f"{ep_r_h:.3f}",
                info["pellets_collected"], info["times_caught"], winner,
                k_updates_this_ep, h_updates_this_ep,
                f"{krishna.last_loss or 0.0:.5f}", f"{hunter.last_loss or 0.0:.5f}",
                f"{krishna.last_entropy or 0.0:.5f}", f"{hunter.last_entropy or 0.0:.5f}",
                f"{krishna.last_approx_kl or 0.0:.5f}", f"{hunter.last_approx_kl or 0.0:.5f}",
                f"{krishna.last_clipfrac or 0.0:.5f}", f"{hunter.last_clipfrac or 0.0:.5f}",
                len(pool), f"{avg100:.2f}",
            ])

            if ep % 10 == 0 or ep == episodes:
                log_file.flush()
                now = time.time()
                ep_per_sec = 10.0 / max(1e-6, (now - last_print_t))
                last_print_t = now
                print(f"  ep {ep:>5}/{episodes}  mode={mode}  steps={ep_steps:>4}  "
                      f"r_k={ep_r_k:>8.2f}  r_h={ep_r_h:>8.2f}  pellets={info['pellets_collected']}  "
                      f"winner={winner:<8}  k_upd={krishna.update_step:>4}  h_upd={hunter.update_step:>4}  "
                      f"pool={len(pool):>2}  avg100={avg100:.2f}  ({ep_per_sec:.2f} ep/s)", flush=True)

            if ep >= episodes:
                break

            mode, hunter_actor, hunter_learns = _pick_mode()
            ep_seed = int(rng.integers(0, 2**31 - 1))
            state, info = env.reset(seed=ep_seed)
            k_prev_done = True
            h_prev_done = True
            ep_r_k = ep_r_h = 0.0
            ep_steps = 0
            k_updates_this_ep = h_updates_this_ep = 0

            record_this = (ep + 1) % max(1, replay_every) == 0 or (ep + 1) == episodes
            if record_this:
                recorder.start_episode(
                    episode_id=ep + 1, phase=0 if mode == "joint" else 1, difficulty=0,
                    seed=ep_seed, grid_size=env.grid_size, agents=["krishna", "hunter"],
                )

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
        "name": f"v8 IPPO self-play — {name}",
        "phase_label": "v8_ippo_selfplay",
        "algo": {
            "name": "Independent PPO (clipped surrogate, GAE) — self-play, FSP",
            "network": "CNNActorCritic(conv=[32,64], fc=256) x2 — same conv trunk as CNNDuelingQNetwork",
            "state_encoding": "6-channel (6,25,25) binary",
            "fsp": {"p_latest_joint": p_latest, "snapshot_every": snapshot_every,
                    "pool_max_size": 20},
        },
        "hyperparams": {
            "learning_rate": 2.5e-4, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_coef": 0.1, "ent_coef": 0.01, "vf_coef": 0.5,
            "max_grad_norm": 0.5, "update_epochs": 4, "num_minibatches": 4,
            "rollout_len": rollout_len,
        },
        "env": {
            "type": "SelfPlayGridworld", "grid_size": env.grid_size,
            "pellets": env.TARGET_PELLETS, "lives": env.KRISHNA_LIVES,
            "max_steps": env.MAX_STEPS,
        },
        "training": {
            "total_episodes": episodes,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration,
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "results": {
            "wins": wins,
            "mode_counts": mode_counts,
            "krishna_win_rate": wins["krishna"] / max(1, episodes),
            "hunter_win_rate": wins["hunter"] / max(1, episodes),
            "timeout_rate": wins["timeout"] / max(1, episodes),
            "final_krishna_updates": krishna.update_step,
            "final_hunter_updates": hunter.update_step,
            "final_pool_size": len(pool),
        },
        "artifacts": {
            "log_csv": "logs/episode_stats.csv",
            "krishna_checkpoint": "checkpoints/krishna_final.pth",
            "hunter_checkpoint": "checkpoints/hunter_final.pth",
            "replays_dir": "replays",
            "pool_dir": "pool",
        },
        "notes": "v8: same SelfPlayGridworld env/reward/FSP-pool design as Phase 2, "
                 "algorithm swapped DQN -> Independent PPO. See Q6.md section 3.3.",
    }
    (run_dir / "experiment.json").write_text(json.dumps(experiment, indent=2))
    print(f"\nWrote {run_dir / 'experiment.json'}")
    print(f"  krishna_win={wins['krishna']}  hunter_win={wins['hunter']}  "
          f"timeout={wins['timeout']}  k_updates={krishna.update_step}  "
          f"h_updates={hunter.update_step}  duration={duration:.1f}s")
    return run_dir


# ----------------------------- cli -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--smoke", action="store_true",
                   help="Run a handful of episodes with a small rollout length "
                        "for pipeline validation (exercises the update path).")
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="v8_ippo_selfplay")
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--p-latest", type=float, default=0.7,
                   help="P(joint training episode); rest are FSP vs frozen.")
    p.add_argument("--replay-every", type=int, default=25)
    p.add_argument("--rollout-len", type=int, default=2048,
                   help="On-policy rollout length (steps) per PPO update.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    episodes = 20 if args.smoke else args.episodes
    rollout_len = 128 if args.smoke else args.rollout_len
    if args.smoke:
        args.snapshot_every = 5
        args.replay_every = 5
    run_training(
        episodes=episodes,
        device=args.device,
        seed=args.seed,
        name=args.name,
        snapshot_every=args.snapshot_every,
        p_latest=args.p_latest,
        replay_every=args.replay_every,
        rollout_len=rollout_len,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
