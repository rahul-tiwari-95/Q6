"""
Quick Training Test - 50 Episodes
===================================
This script runs a quick 50-episode training session to verify the training loop works.
Much faster than full 2000 episodes, lets us verify the fix immediately.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from collections import deque

# Import our custom modules
from environment.hunter_gridworld import HunterGridworld
from agent.dq_agent import DQNAgent
from utils.logger import TrainingLogger
from utils.environment_wrapper import EnvironmentMetricsWrapper


def quick_train(n_episodes: int = 50):
    """Quick training for verification."""
    
    print("\n" + "="*80)
    print(f"QUICK TRAINING TEST - {n_episodes} EPISODES")
    print("="*80 + "\n")
    
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
    
    # Training tracking variables
    scores = []
    scores_window = deque(maxlen=100)
    
    print(f"Running {n_episodes} episodes...\n")
    
    # Main training loop
    for i_episode in range(1, n_episodes + 1):
        # Reset environment for new episode
        state, info = env.reset()
        score = 0
        done = False
        
        # Episode loop
        for t in range(1000):
            action = agent.act(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done or truncated)
            
            score += reward
            state = next_state
            
            if done or truncated:
                break
        
        # Track scores
        metrics = env.get_episode_metrics()
        scores_window.append(score)
        scores.append(score)
        
        # Progress
        print(f'\r[Episode {i_episode:2d}/{n_episodes}] '
              f'Score: {score:8.2f} | '
              f'Avg(100): {np.mean(scores_window):8.2f} | '
              f'ε: {agent.epsilon:.4f}', end='')
    
    print("\n\n" + "="*80)
    print("QUICK TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes:     {len(scores)}")
    print(f"Avg score (all):    {np.mean(scores):.2f}")
    print(f"Avg score (last 10): {np.mean(scores[-10:]):.2f}")
    print(f"Max score:          {max(scores):.2f}")
    print(f"Min score:          {min(scores):.2f}")
    print(f"Final epsilon:      {agent.epsilon:.4f}")
    print("="*80 + "\n")
    
    return scores


if __name__ == '__main__':
    scores = quick_train(n_episodes=50)
    print("✓ Training loop works correctly!")
    print(f"✓ Completed {len(scores)} episodes successfully")
    print("\nNow you can run: python main.py")
