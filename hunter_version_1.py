"""
Main Training Script - DQN Agent Training in HunterGridworld
============================================================
This is the primary training orchestrator for Kṛṣṇa.

What this script does:
1. Initializes the environment and agent
2. Runs training episodes
3. Logs detailed information about every step and episode
4. Saves checkpoints and final statistics
5. Visualizes learning progress

Think of this as the "command center" that oversees the entire learning process.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import our custom modules
from environment.hunter_gridworld import HunterGridworld
from agent.dq_agent import DQNAgent
from utils.logger import TrainingLogger
from utils.environment_wrapper import EnvironmentMetricsWrapper


def action_to_name(action: int) -> str:
    """
    Convert action integer to human-readable name.
    
    Args:
        action (int): Action code (0=Up, 1=Down, 2=Left, 3=Right)
    Returns:
        str: Human-readable action name
    """
    actions = {0: "UP   ", 1: "DOWN ", 2: "LEFT ", 3: "RIGHT"}
    return actions.get(action, "UNKNOWN")


def create_training_directories() -> Tuple[str, str, str]:
    """
    Create directories for saving training artifacts.
    
    We organize training runs by timestamp so we can compare different runs.
    
    Returns:
        tuple: Paths to checkpoint, log, and plot directories
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('training_runs', timestamp)
    
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    log_dir = os.path.join(base_dir, 'logs')
    plot_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("TRAINING DIRECTORIES CREATED")
    print(f"{'='*80}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Log Dir:        {log_dir}")
    print(f"Plot Dir:       {plot_dir}")
    print(f"{'='*80}\n")
    
    return checkpoint_dir, log_dir, plot_dir


def plot_scores(scores: List[float], 
                avg_scores: List[float], 
                save_path: str,
                episode: int) -> None:
    """
    Plot training scores and save the figure.
    
    Visualizing training progress helps us see if the agent is learning or not.
    
    Args:
        scores: List of episode scores
        avg_scores: List of moving average scores
        save_path: Path to save the plot
        episode: Current episode number
    """
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Episode Score', alpha=0.5, linewidth=1)
    plt.plot(avg_scores, label='100-Episode Moving Average', linewidth=2, color='red')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Training Progress (Episodes 1-{episode})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def print_hyperparameters(agent: DQNAgent) -> None:
    """
    Print the hyperparameters being used for training.
    
    It's important to log these so we know what settings produced what results.
    
    Args:
        agent (DQNAgent): The agent to print hyperparameters for
    """
    print(f"\n{'='*80}")
    print("HYPERPARAMETERS")
    print(f"{'='*80}")
    print(f"State Size:              {agent.state_size}")
    print(f"Action Size:             {agent.action_size}")
    print(f"Hidden Layer Sizes:      {agent.hidden_sizes}")
    print(f"Learning Rate (α):       {agent.learning_rate}")
    print(f"Discount Factor (γ):     {agent.gamma}")
    print(f"Soft Update (τ):         {agent.tau}")
    print(f"Batch Size:              {agent.batch_size}")
    print(f"Replay Buffer Size:      {agent.memory.buffer.maxlen}")
    print(f"Device:                  {agent.device}")
    print(f"Epsilon Start:           {agent.epsilon:.4f}")
    print(f"Epsilon Min:             {agent.epsilon_min:.4f}")
    print(f"Epsilon Decay:           {agent.epsilon_decay:.4f}")
    print(f"{'='*80}\n")


def print_environment_info(env: HunterGridworld) -> None:
    """
    Print information about the environment.
    
    Args:
        env (HunterGridworld): The environment to print info about
    """
    print(f"\n{'='*80}")
    print("ENVIRONMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Grid Size:               25x25")
    print(f"State Space Size:        {env.state_size} (flattened)")
    print(f"Number of Entities:      7 (1 Player + 5 Enemies + Walls/Pellets)")
    print(f"  - Kṛṣṇa (Player):      1")
    print(f"  - Hunter:              1 (Aggressive, uses pathfinding)")
    print(f"  - Greedy Bots:         2 (Chase pellets)")
    print(f"  - Patrollers:          2 (Patrol fixed routes)")
    print(f"Actions:                 4 (Up, Down, Left, Right)")
    print(f"\nReward Structure:")
    print(f"  Collect Pellet:        +50")
    print(f"  Get Caught:            -20")
    print(f"  Hit Wall:              -5")
    print(f"  Per Step Penalty:      -0.01 (changed from -0.1 for better learning)")
    print(f"  Win (Score >= 200):    +100")
    print(f"\nWin Condition:           Collect 4 pellets (4 × 50 = 200 points)")
    print(f"Loss Condition:          Lose all 3 lives")
    print(f"{'='*80}\n")


def train(n_episodes: int = 2000,
          max_steps: int = 1000,
          verbose_every: int = 10) -> None:
    """
    Main training function.
    
    This orchestrates the entire training process:
    1. Sets up logging infrastructure
    2. Initializes environment and agent
    3. Runs training loop
    4. Periodically saves checkpoints and visualizations
    5. Reports statistics
    
    Args:
        n_episodes (int): Number of episodes to train
        max_steps (int): Maximum steps per episode
        verbose_every (int): Print detailed logs every N episodes
    """
    
    print("\n" + "="*80)
    print("INITIALIZING TRAINING")
    print("="*80)
    
    # Create directories for artifacts
    checkpoint_dir, log_dir, plot_dir = create_training_directories()
    
    # Initialize logger
    logger = TrainingLogger(log_dir)
    
    # Initialize wrapped environment and agent
    base_env = HunterGridworld()
    env = EnvironmentMetricsWrapper(base_env)
    agent = DQNAgent(
        state_size=625,
        action_size=4,
        hidden_sizes=(256, 128),
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=64
    )
    
    # Print configuration info
    print_environment_info(base_env)
    print_hyperparameters(agent)
    
    # Training tracking variables
    scores = []
    scores_window = deque(maxlen=100)
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    # ===== MAIN TRAINING LOOP =====
    for i_episode in range(1, n_episodes + 1):
        # Reset environment for new episode
        state, info = env.reset()
        score = 0
        done = False
        
        # Track action distribution for this episode
        actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # ===== EPISODE LOOP =====
        for t in range(max_steps):
            # Agent chooses action using epsilon-greedy policy
            # With probability epsilon: random action (exploration)
            # With probability 1-epsilon: best action (exploitation)
            action = agent.act(state, training=True)
            actions_taken[action] += 1
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience in replay buffer and learn
            agent.step(state, action, reward, next_state, done or truncated)
            
            score += reward
            state = next_state
            
            # Print very detailed step-by-step info if verbose
            if (i_episode % verbose_every == 0 and t < 5):
                print(f"  Step {t+1}: Action={action_to_name(action)} "
                      f"Position={info.get('krishna_position', 'N/A')} "
                      f"Reward={reward:6.2f} Score={info.get('score', 0)} "
                      f"Lives={info.get('lives', 3)}")
            
            if done or truncated:
                break
        
        # ===== END OF EPISODE =====
        
        # Get episode metrics
        metrics = env.get_episode_metrics()
        scores_window.append(score)
        scores.append(score)
        
        # Log episode
        logger.log_episode_end(
            episode=i_episode,
            total_reward=score,
            total_steps=metrics['total_steps'],
            pellets_collected=metrics['pellets_collected'],
            times_caught=metrics['times_caught'],
            walls_hit=metrics['walls_hit'],
            won=agent.krishna.score >= 200 if hasattr(agent, 'krishna') else False,
            epsilon=agent.epsilon,
            avg_reward_per_step=metrics['avg_reward_per_step']
        )
        
        # Print progress indicator (one per episode)
        print(f'\r[Episode {i_episode:4d}/{n_episodes}] '
              f'Score: {score:8.2f} | '
              f'Avg(100): {np.mean(scores_window):8.2f} | '
              f'ε: {agent.epsilon:.4f} | '
              f'Collected: {metrics["pellets_collected"]} | '
              f'Caught: {metrics["times_caught"]}', end='')
        
        # Detailed logging every verbose_every episodes
        if i_episode % verbose_every == 0:
            print()  # New line
            logger.print_episode_summary(i_episode, list(scores_window), agent.epsilon)
            
            # Print action distribution
            print(f"\nAction Distribution in Episode {i_episode}:")
            action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
            for action_idx, count in actions_taken.items():
                pct = (count / max(1, metrics['total_steps'])) * 100
                print(f"  {action_names[action_idx]}: {count:4d} times ({pct:5.1f}%)")
        
        # Save checkpoint and plot every 100 episodes
        if i_episode % 100 == 0:
            # Save agent checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i_episode}.pth')
            agent.save(checkpoint_path)
            
            # Create and save plot
            avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
            plot_path = os.path.join(plot_dir, f'training_progress_{i_episode}.png')
            plot_scores(scores, avg_scores, plot_path, i_episode)
            
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")
            print(f"✓ Plot saved: {plot_path}\n")
            
        # Early stopping if environment is solved
        # NOTE: Only stop if we have 100+ episodes AND sustained high performance
        # This prevents stopping after just 1-2 lucky episodes
        if i_episode >= 100 and np.mean(scores_window) >= 200.0:
            print(f'\n\n{"="*80}')
            print(f'Environment solved in {i_episode} episodes!')
            print(f'Moving average score: {np.mean(scores_window):.2f}')
            print(f'{"="*80}\n')
            
            # Save final checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            agent.save(checkpoint_path)
            break
    
    # ===== TRAINING COMPLETE =====
    
    # Create final plot
    print("\nGenerating final plots...")
    avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plot_path = os.path.join(plot_dir, 'final_training_progress.png')
    plot_scores(scores, avg_scores, plot_path, len(scores))
    
    # Save final statistics
    logger.save_final_stats()
    
    print(f"\nTraining complete! Check {log_dir} for detailed logs.")


if __name__ == '__main__':
    train()
