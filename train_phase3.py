"""
Phase 3: Self-play training with Gated Option Policies and Counterfactual HER.

New features over train_phase2.py
-----------------------------------
* GatedDQNAgent for Krishna — dual evade/collect heads + learned context gate.
* Counterfactual HER — injects safe-collect opportunities from completed
  episodes into Krishna's buffer (addresses collection paralysis).
* HierarchicalOpponentPool — 5 easy (permanent) + 15 hard (FIFO) tiers so
  Krishna always has a weak opponent to practice collection against.
* Best-checkpoint saving — saves whenever avg-100 Krishna reward improves.
* Gate diagnostics — logs mean gate value per episode to CSV.

Usage
-----
    # Quick smoke test (10 episodes)
    python train_phase3.py --smoke

    # Full run warm-starting from v4 checkpoint
    python train_phase3.py \\
        --episodes 12000 \\
        --device mps \\
        --name phase3_v1_gop_cher \\
        --warm-start training_runs/20260606_175631_phase2_v4_nowall/checkpoints/krishna_best_ep5300.pth

    # Start fresh
    python train_phase3.py --episodes 12000 --device mps --name phase3_v1_fresh
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
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

from agent.gated_dqn_agent import GatedDQNAgent, info_to_context
from agent.dqn_v2_agent import DQNv2Agent
from agent.frozen_agent import FrozenAgent
from utils.hierarchical_pool import HierarchicalOpponentPool
from utils.cher import CounterfactualHER
from config import GRID_SIZE
from environment.selfplay_env import SelfPlayGridworld
from utils.replay_recorder import ReplayRecorder


REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _new_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = REPO_ROOT / "training_runs" / f"{ts}_{name}"
    for sub in ("checkpoints", "logs", "replays", "pool"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _save_checkpoint_state(
    run_dir: Path,
    episode: int,
    krishna,
    hunter,
    best_avg100: float,
    best_ckpt_ep: int,
    wins: dict,
    mode_counts: dict,
    rolling: list,
    total_episodes: int,
    name: str,
    started_at: str,
) -> None:
    """Persist enough state to resume training from this episode."""
    krishna.save(str(run_dir / "checkpoints" / "krishna_latest.pth"))
    hunter.save(str(run_dir / "checkpoints" / "hunter_latest.pth"))
    payload = {
        "episode":         episode,
        "total_episodes":  total_episodes,
        "name":            name,
        "krishna_eps":     krishna.epsilon,
        "hunter_eps":      hunter.epsilon,
        "best_avg100":     best_avg100,
        "best_avg100_ep":  best_ckpt_ep,
        "wins":            wins,
        "mode_counts":     mode_counts,
        "rolling_rewards": list(rolling),
        "started_at":      started_at,
    }
    (run_dir / "checkpoints" / "last_state.json").write_text(
        json.dumps(payload, indent=2)
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    episodes: int,
    device: str | None,
    seed: int,
    name: str,
    snapshot_every: int,
    p_latest: float,
    replay_every: int,
    warm_start: str | None,
    cher_safe_dist: int,
    cher_pellet_reach: int,
    cher_bonus: float,
    epsilon_start: float,
    resume_dir: str | None = None,
    checkpoint_every: int = 100,
    easy_warmup_eps: int = 1000,
) -> Path:
    # --- Run directory: new or resumed ---
    saved_state: dict | None = None
    if resume_dir:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        state_path = run_dir / "checkpoints" / "last_state.json"
        if not state_path.exists():
            raise FileNotFoundError(
                f"No checkpoint state in {resume_dir}.\n"
                "Training must have been started with --checkpoint-every (default is 100)."
            )
        saved_state = json.loads(state_path.read_text())
        ep_start = saved_state["episode"] + 1
        print(f"[resume] Resuming from ep {ep_start}/{episodes}  run_dir={run_dir}")
    else:
        run_dir  = _new_run_dir(name)
        ep_start = 1

    rng = np.random.default_rng(seed + (ep_start - 1))
    torch.manual_seed(seed + (ep_start - 1))

    env = SelfPlayGridworld(grid_size=GRID_SIZE)

    # --- Agents ---
    krishna = GatedDQNAgent(device=device)
    hunter  = DQNv2Agent(device=device)

    if resume_dir and saved_state:
        print("[resume] Loading Krishna from checkpoints/krishna_latest.pth")
        krishna.load(str(run_dir / "checkpoints" / "krishna_latest.pth"))
        print("[resume] Loading Hunter  from checkpoints/hunter_latest.pth")
        hunter.load(str(run_dir / "checkpoints" / "hunter_latest.pth"))
    elif warm_start:
        print(f"[warm-start] loading Krishna from {warm_start}")
        krishna.load_warm_start(warm_start)
        krishna.reset_epsilon(epsilon_start)
        print(f"[warm-start] epsilon reset to {epsilon_start}")
    else:
        krishna.reset_epsilon(epsilon_start)

    # --- Opponent pool (hierarchical) ---
    # Pool snapshots are persisted on disk — on resume they reload automatically.
    pool = HierarchicalOpponentPool(
        run_dir / "pool", easy_max=5, hard_max=15
    )
    if not resume_dir:
        pool.add_snapshot(hunter, metadata={"episode": 0, "kind": "init"})

    # --- CHER ---
    cher = CounterfactualHER(
        hunter_safe_dist=cher_safe_dist,
        pellet_reach=cher_pellet_reach,
        collect_bonus=cher_bonus,
    )

    # --- Replay recorder ---
    recorder = ReplayRecorder(str(run_dir), enabled=True)

    # --- CSV log (append on resume, fresh otherwise) ---
    log_path = run_dir / "logs" / "episode_stats.csv"
    log_file = log_path.open("a" if resume_dir else "w", newline="")
    log = csv.writer(log_file)
    if not resume_dir:
        log.writerow([
            "episode", "mode", "steps",
            "krishna_reward", "hunter_reward",
            "pellets", "caught", "winner",
            "krishna_eps", "hunter_eps",
            "krishna_loss", "hunter_loss",
            "pool_size", "cf_injected",
            "cher_opportunities", "cher_dep_idx",
            "mean_gate", "avg100",
        ])

    now_ts     = datetime.now(timezone.utc).isoformat()
    started_at = saved_state["started_at"] if saved_state else now_ts
    t0 = time.time()
    last_print_t = t0

    # --- Restore or initialise rolling stats ---
    if saved_state:
        wins         = saved_state["wins"]
        mode_counts  = saved_state["mode_counts"]
        rolling      = list(saved_state["rolling_rewards"])
        best_avg100  = saved_state["best_avg100"]
        best_ckpt_ep = saved_state["best_avg100_ep"]
    else:
        wins         = {"krishna": 0, "hunter": 0, "timeout": 0}
        mode_counts  = {"joint": 0, "fsp": 0}
        best_avg100  = float("-inf")
        best_ckpt_ep = 0
        rolling: list[float] = []

    print(
        f"[start] {episodes} eps  device={krishna.device}  "
        f"p_latest={p_latest}  snapshot_every={snapshot_every}  "
        f"cher_safe={cher_safe_dist}  cher_reach={cher_pellet_reach}  "
        f"cher_bonus={cher_bonus}  run_dir={run_dir}",
        flush=True,
    )

    # --- Graceful shutdown: save checkpoint on SIGTERM / SIGINT ---
    def _on_shutdown(signum, frame):
        print(f"\n[checkpoint] signal {signum} — saving at ep {ep}...", flush=True)
        try:
            _save_checkpoint_state(
                run_dir, ep, krishna, hunter,
                best_avg100, best_ckpt_ep,
                wins, mode_counts, rolling,
                episodes, name, started_at,
            )
            print("[checkpoint] saved  →  checkpoints/last_state.json", flush=True)
        except Exception as exc:
            print(f"[checkpoint] save failed: {exc}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_shutdown)
    signal.signal(signal.SIGINT, _on_shutdown)

    for ep in range(ep_start, episodes + 1):
        mode = "joint" if rng.random() < p_latest else "fsp"
        mode_counts[mode] += 1

        if mode == "fsp":
            # Two-phase easy curriculum: use easy opponents exclusively for warm-up
            p_easy_cur = 1.0 if ep <= easy_warmup_eps else 0.25
            snap = pool.sample(rng, p_easy=p_easy_cur, p_latest=0.3)
            hunter_actor = FrozenAgent.load(snap, device=krishna.device)
            hunter_learns = False
        else:
            hunter_actor = hunter
            hunter_learns = True

        ep_seed = int(rng.integers(0, 2**31 - 1))
        state, info = env.reset(seed=ep_seed)
        context = info_to_context(info, GRID_SIZE)

        record_this = (ep % max(1, replay_every) == 0) or (ep == episodes)
        if record_this:
            recorder.start_episode(
                episode_id=ep, phase=0 if mode == "joint" else 1,
                difficulty=0, seed=ep_seed,
                grid_size=env.grid_size, agents=["krishna", "hunter"],
            )

        ep_r_k = 0.0
        ep_r_h = 0.0
        steps   = 0
        done = trunc = False
        trajectory: list[dict] = []

        while not (done or trunc):
            a_k = krishna.act(state, context, training=True)
            a_h = hunter_actor.act(state, training=True)

            next_state, rewards, done, trunc, info = env.step(
                {"krishna": a_k, "hunter": a_h}
            )
            next_context = info_to_context(info, GRID_SIZE)

            # --- Store experience + learn ---
            krishna.step(
                state, context, a_k, rewards["krishna"],
                next_state, next_context, done or trunc,
            )
            if hunter_learns:
                hunter.step(
                    state, a_h, rewards["hunter"], next_state, done or trunc
                )

            # --- Accumulate trajectory for CHER ---
            trajectory.append({
                "state":        state,
                "context":      context,
                "action":       a_k,
                "reward":       rewards["krishna"],
                "next_state":   next_state,
                "next_context": next_context,
                "done":         done or trunc,
                "info":         info,
            })

            if record_this:
                opts = krishna.q_values_options(state, context)
                recorder.record_step(
                    t=steps,
                    grid=env.grid.tolist(),
                    actions={"krishna": int(a_k), "hunter": int(a_h)},
                    rewards={
                        "krishna": float(rewards["krishna"]),
                        "hunter":  float(rewards["hunter"]),
                    },
                    q_values={
                        "krishna": opts["q_blend"].tolist(),
                        "hunter":  hunter_actor.q_values(state).tolist(),
                    },
                    lives=info["lives"],
                    pellets=info["pellets_collected"],
                    done=bool(done or trunc),
                )

            state   = next_state
            context = next_context
            ep_r_k += rewards["krishna"]
            ep_r_h += rewards["hunter"]
            steps  += 1

        winner = info.get("winner") or "timeout"
        wins[winner] = wins.get(winner, 0) + 1

        if record_this:
            recorder.end_episode(
                outcome=winner,
                total_reward={"krishna": float(ep_r_k), "hunter": float(ep_r_h)},
                pellets_collected=info["pellets_collected"],
            )

        # --- CHER injection ---
        cf_exps          = cher.relabel(trajectory)
        cf_injected      = len(cf_exps)
        cher_opps        = cher.count_opportunities(trajectory)
        cher_dep_idx     = cf_injected / max(1, cher_opps)
        if cf_exps:
            krishna.memory.add_batch(cf_exps)

        # --- Epsilon decay ---
        krishna.decay_epsilon()
        if hunter_learns:
            hunter.decay_epsilon()

        # --- FSP snapshot ---
        if ep % snapshot_every == 0:
            pool.add_snapshot(hunter, metadata={"episode": ep})

        # --- Periodic resume checkpoint ---
        if checkpoint_every > 0 and ep % checkpoint_every == 0:
            _save_checkpoint_state(
                run_dir, ep, krishna, hunter,
                best_avg100, best_ckpt_ep,
                wins, mode_counts, rolling,
                episodes, name, started_at,
            )

        # --- Rolling average and best checkpoint ---
        rolling.append(ep_r_k)
        if len(rolling) > 100:
            rolling.pop(0)
        avg100 = float(np.mean(rolling))

        if avg100 > best_avg100 and ep >= 100:
            best_avg100  = avg100
            best_ckpt_ep = ep
            krishna.save(str(run_dir / "checkpoints" / "krishna_best.pth"))

        mean_gate = krishna.last_mean_gate or 0.0

        log.writerow([
            ep, mode, steps,
            f"{ep_r_k:.3f}", f"{ep_r_h:.3f}",
            info["pellets_collected"], info["times_caught"], winner,
            f"{krishna.epsilon:.4f}", f"{hunter.epsilon:.4f}",
            f"{krishna.last_loss or 0.0:.5f}",
            f"{hunter.last_loss or 0.0:.5f}",
            len(pool), cf_injected,
            cher_opps, f"{cher_dep_idx:.3f}",
            f"{mean_gate:.4f}", f"{avg100:.2f}",
        ])

        if ep % 10 == 0 or ep == episodes:
            log_file.flush()
            now       = time.time()
            ep_rate   = 10.0 / max(1e-6, now - last_print_t) if ep > 0 else 0.0
            last_print_t = now
            print(
                f"  ep {ep:>5}/{episodes}  mode={mode}  steps={steps:>4}  "
                f"r_k={ep_r_k:>8.2f}  pellets={info['pellets_collected']}  "
                f"winner={winner:<8}  pool={len(pool):>2}  "
                f"avg100={avg100:>7.2f}  gate={mean_gate:.3f}  "
                f"cf={cf_injected:>3}  ({ep_rate:.2f} ep/s)",
                flush=True,
            )

    # --- Finalise ---
    log_file.close()
    recorder.close()

    krishna.save(str(run_dir / "checkpoints" / "krishna_final.pth"))
    hunter.save(str(run_dir / "checkpoints" / "hunter_final.pth"))

    ended_at = datetime.now(timezone.utc).isoformat()
    duration = time.time() - t0

    experiment = {
        "id":          run_dir.name,
        "name":        f"Phase3 GOP+CHER — {name}",
        "phase_label": "phase3_gop_cher",
        "algo": {
            "name":     "Gated Option DQN + Counterfactual HER + Hierarchical FSP",
            "network":  "GatedOptionNetwork(evade+collect heads, gate_mlp(3->16->1))",
            "state_encoding": "6-channel (6,25,25) binary",
            "loss":     "smooth_l1",
            "cher": {
                "hunter_safe_dist": cher_safe_dist,
                "pellet_reach":     cher_pellet_reach,
                "collect_bonus":    cher_bonus,
            },
            "fsp": {
                "p_latest_joint":  p_latest,
                "snapshot_every":  snapshot_every,
                "easy_tier_slots": 5,
                "hard_tier_slots": 15,
            },
        },
        "hyperparams": {
            "learning_rate": 1e-4, "gamma": 0.99, "tau": 0.001,
            "batch_size": 64, "buffer_size": 100_000, "update_every": 4,
            "epsilon_start": epsilon_start, "epsilon_min": 0.05,
            "epsilon_decay": 0.9994,
        },
        "env": {
            "type":      "SelfPlayGridworld",
            "grid_size": env.grid_size,
            "pellets":   env.TARGET_PELLETS,
            "lives":     env.KRISHNA_LIVES,
            "max_steps": env.MAX_STEPS,
        },
        "training": {
            "total_episodes": episodes,
            "warm_start":     warm_start or "none",
            "started_at":     started_at,
            "ended_at":       ended_at,
            "duration_seconds": duration,
            "git_branch":     _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_commit":     _git("rev-parse", "HEAD"),
            "git_dirty":      bool(_git("status", "--porcelain")),
            "easy_warmup_eps": easy_warmup_eps,
        },
        "results": {
            "wins":              wins,
            "mode_counts":       mode_counts,
            "krishna_win_rate":  wins["krishna"] / max(1, episodes),
            "hunter_win_rate":   wins["hunter"]  / max(1, episodes),
            "timeout_rate":      wins["timeout"] / max(1, episodes),
            "best_avg100":       best_avg100,
            "best_avg100_ep":    best_ckpt_ep,
            "final_epsilon_k":   krishna.epsilon,
            "final_epsilon_h":   hunter.epsilon,
            "final_pool_size":   len(pool),
        },
        "artifacts": {
            "log_csv":               "logs/episode_stats.csv",
            "krishna_best":          "checkpoints/krishna_best.pth",
            "krishna_final":         "checkpoints/krishna_final.pth",
            "hunter_final":          "checkpoints/hunter_final.pth",
            "replays_dir":           "replays",
            "pool_dir":              "pool",
        },
        "notes": "",
    }
    (run_dir / "experiment.json").write_text(json.dumps(experiment, indent=2))

    print(f"\n[done] {run_dir}")
    print(f"  krishna_win={wins['krishna']}  hunter_win={wins['hunter']}  "
          f"timeout={wins['timeout']}  best_avg100={best_avg100:.2f} @ ep {best_ckpt_ep}  "
          f"duration={duration:.1f}s")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3: GOP + CHER self-play training")
    p.add_argument("--episodes",         type=int,   default=None,
                   help="Total episodes (default 12000; on --resume uses saved value)")
    p.add_argument("--device",           type=str,   default=None,
                   help="cuda | mps | cpu  (auto-detected if omitted)")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--name",             type=str,   default="phase3_gop_cher")
    p.add_argument("--snapshot-every",   type=int,   default=100)
    p.add_argument("--p-latest",         type=float, default=0.7,
                   help="Prob of joint-training episode (default 0.7)")
    p.add_argument("--replay-every",     type=int,   default=25,
                   help="Record full JSONL replay every N episodes")
    p.add_argument("--warm-start",       type=str,   default=None,
                   help="Path to DQNv2Agent checkpoint for warm-starting Krishna")
    p.add_argument("--epsilon-start",    type=float, default=0.15,
                   help="Initial epsilon for Krishna (default 0.15 for warm-start)")
    p.add_argument("--cher-safe-dist",   type=int,   default=6)
    p.add_argument("--cher-pellet-reach",type=int,   default=4)
    p.add_argument("--cher-bonus",       type=float, default=30.0)
    p.add_argument("--resume",           type=str,   default=None,
                   help="Path to existing run_dir to resume (e.g. training_runs/20260607_…)")
    p.add_argument("--checkpoint-every", type=int,   default=100,
                   help="Save resumable checkpoint every N episodes (0 = disable)")
    p.add_argument("--smoke",            action="store_true",
                   help="Smoke test: 10 episodes only")
    p.add_argument("--easy-warmup-eps",  type=int,   default=1000,
                   help="Episodes to use p_easy=1.0 before switching to normal sampling (default 1000)")
    args = p.parse_args()

    # Resolve total episode count
    if args.resume and args.episodes is None:
        state_path = Path(args.resume) / "checkpoints" / "last_state.json"
        saved = json.loads(state_path.read_text())
        episodes = saved["total_episodes"]
    else:
        episodes = args.episodes if args.episodes is not None else 12_000

    if args.smoke:
        episodes          = 10
        args.replay_every = 1
        print("[smoke] 10 episodes")

    run_training(
        episodes          = episodes,
        device            = args.device,
        seed              = args.seed,
        name              = args.name,
        snapshot_every    = args.snapshot_every,
        p_latest          = args.p_latest,
        replay_every      = args.replay_every,
        warm_start        = args.warm_start,
        cher_safe_dist    = args.cher_safe_dist,
        cher_pellet_reach = args.cher_pellet_reach,
        cher_bonus        = args.cher_bonus,
        epsilon_start     = args.epsilon_start,
        resume_dir        = args.resume,
        checkpoint_every  = args.checkpoint_every,
        easy_warmup_eps   = args.easy_warmup_eps,
    )


if __name__ == "__main__":
    main()
