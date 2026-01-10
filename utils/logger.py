"""
Logging and Monitoring System
-----------------------------
This module provides comprehensive logging and monitoring for training.
It helps us understand exactly what the agent is doing at every step.

Think of this as a detailed "training journal" that records:
- What action the agent takes
- What reward it gets
- How the enemies move
- When the agent collides or collects pellets
- Training statistics and trends
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class TrainingLogger:
    """
    Comprehensive logger for tracking training progress and agent behavior.
    
    This logger records:
    1. Per-step information (detailed action, reward, state)
    2. Per-episode summaries (total reward, pellets collected, enemies encountered)
    3. Training-wide statistics (learning curves, convergence metrics)
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the training logger.
        
        Args:
            log_dir (str): Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Per-step information storage
        self.current_episode_steps = []
        
        # Per-episode summaries
        self.episode_summaries = []
        
        # Training statistics
        self.training_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'pellets_collected_total': 0,
            'times_caught_total': 0,
            'walls_hit_total': 0,
            'episodes_won': 0,
            'epsilon_history': []
        }
        
        # Initialize log files
        self.episode_log_file = os.path.join(log_dir, 'episodes.jsonl')
        self.training_log_file = os.path.join(log_dir, 'training.log')
        self.stats_file = os.path.join(log_dir, 'stats.json')
        
    def log_step(self, 
                 episode: int,
                 step: int,
                 state_position: tuple,
                 action: int,
                 action_name: str,
                 reward: float,
                 next_state_position: tuple,
                 done: bool,
                 info: Dict[str, Any],
                 epsilon: float) -> None:
        """
        Log detailed information about a single step in an episode.
        
        This captures the micro-level details of what's happening.
        
        Args:
            episode (int): Episode number
            step (int): Step number within episode
            state_position (tuple): Agent's position before action
            action (int): Action taken (0=Up, 1=Down, 2=Left, 3=Right)
            action_name (str): Human-readable action name
            reward (float): Reward received
            next_state_position (tuple): Agent's position after action
            done (bool): Whether episode ended
            info (Dict): Additional information
            epsilon (float): Current exploration rate
        """
        step_info = {
            'episode': episode,
            'step': step,
            'position_before': state_position,
            'position_after': next_state_position,
            'action': action,
            'action_name': action_name,
            'reward': reward,
            'epsilon': epsilon,
            'score': info.get('score', 0),
            'lives': info.get('lives', 3),
            'pellets_remaining': info.get('pellets_remaining', 0),
            'is_invulnerable': info.get('is_invulnerable', False),
            'episode_ended': done
        }
        
        self.current_episode_steps.append(step_info)
    
    def log_episode_end(self, 
                       episode: int,
                       total_reward: float,
                       total_steps: int,
                       pellets_collected: int,
                       times_caught: int,
                       walls_hit: int,
                       won: bool,
                       epsilon: float,
                       avg_reward_per_step: float) -> None:
        """
        Log summary information at the end of an episode.
        
        This captures the macro-level outcome of an episode.
        
        Args:
            episode (int): Episode number
            total_reward (float): Total reward accumulated in episode
            total_steps (int): Number of steps taken
            pellets_collected (int): Number of pellets collected
            times_caught (int): Number of times caught by enemies
            walls_hit (int): Number of walls hit
            won (bool): Whether agent won the episode
            epsilon (float): Exploration rate at end of episode
            avg_reward_per_step (float): Average reward per step
        """
        episode_summary = {
            'episode': episode,
            'total_reward': total_reward,
            'total_steps': total_steps,
            'pellets_collected': pellets_collected,
            'times_caught': times_caught,
            'walls_hit': walls_hit,
            'won': won,
            'epsilon': epsilon,
            'avg_reward_per_step': avg_reward_per_step,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_summaries.append(episode_summary)
        
        # Update training statistics
        self.training_stats['total_episodes'] += 1
        self.training_stats['total_steps'] += total_steps
        self.training_stats['pellets_collected_total'] += pellets_collected
        self.training_stats['times_caught_total'] += times_caught
        self.training_stats['walls_hit_total'] += walls_hit
        if won:
            self.training_stats['episodes_won'] += 1
        self.training_stats['epsilon_history'].append(epsilon)
        
        # Save episode data to JSONL file (line-delimited JSON)
        with open(self.episode_log_file, 'a') as f:
            f.write(json.dumps(episode_summary) + '\n')
        
        # Clear step data for next episode
        self.current_episode_steps = []
    
    def get_moving_average(self, values: List[float], window: int = 100) -> List[float]:
        """
        Calculate moving average over a window of values.
        
        This smooths out noise in the training signal so we can see trends better.
        
        Args:
            values (List[float]): Values to average
            window (int): Window size for moving average
            
        Returns:
            List[float]: Moving average values
        """
        if len(values) < window:
            return values
        
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            avg = np.mean(values[start_idx:i+1])
            moving_avg.append(avg)
        return moving_avg
    
    def print_episode_summary(self, 
                             episode: int,
                             scores_window: List[float],
                             epsilon: float) -> None:
        """
        Print a human-readable summary of the current episode.
        
        This is what appears on screen during training.
        
        Args:
            episode (int): Episode number
            scores_window (List[float]): Recent episode scores
            epsilon (float): Current exploration rate
        """
        if self.episode_summaries:
            last_episode = self.episode_summaries[-1]
            avg_score = np.mean(scores_window)
            
            print(f"\n{'='*80}")
            print(f"Episode {episode:4d} Summary")
            print(f"{'='*80}")
            print(f"  Total Reward:          {last_episode['total_reward']:10.2f}")
            print(f"  Average Score (100):   {avg_score:10.2f}")
            print(f"  Steps Taken:           {last_episode['total_steps']:10d}")
            print(f"  Pellets Collected:     {last_episode['pellets_collected']:10d}")
            print(f"  Times Caught:          {last_episode['times_caught']:10d}")
            print(f"  Walls Hit:             {last_episode['walls_hit']:10d}")
            print(f"  Avg Reward/Step:       {last_episode['avg_reward_per_step']:10.4f}")
            print(f"  Exploration Rate (ε):  {epsilon:10.4f}")
            print(f"  Episode Won:           {'YES' if last_episode['won'] else 'NO':>10s}")
            print(f"{'='*80}")
    
    def save_final_stats(self) -> None:
        """
        Save final training statistics to JSON file.
        """
        # Calculate overall statistics
        rewards = [ep['total_reward'] for ep in self.episode_summaries]
        steps = [ep['total_steps'] for ep in self.episode_summaries]
        
        final_stats = {
            **self.training_stats,
            'total_episodes': len(self.episode_summaries),
            'average_reward': float(np.mean(rewards)) if rewards else 0,
            'max_reward': float(np.max(rewards)) if rewards else 0,
            'min_reward': float(np.min(rewards)) if rewards else 0,
            'average_steps_per_episode': float(np.mean(steps)) if steps else 0,
            'win_rate': self.training_stats['episodes_won'] / len(self.episode_summaries) if self.episode_summaries else 0,
            'total_pellets_collected': self.training_stats['pellets_collected_total'],
            'total_times_caught': self.training_stats['times_caught_total'],
            'total_walls_hit': self.training_stats['walls_hit_total']
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n{'='*80}")
        print("FINAL TRAINING STATISTICS")
        print(f"{'='*80}")
        print(f"Total Episodes:            {final_stats['total_episodes']}")
        print(f"Episodes Won:              {self.training_stats['episodes_won']}")
        print(f"Win Rate:                  {final_stats['win_rate']*100:.2f}%")
        print(f"Average Reward:            {final_stats['average_reward']:.2f}")
        print(f"Max Reward:                {final_stats['max_reward']:.2f}")
        print(f"Min Reward:                {final_stats['min_reward']:.2f}")
        print(f"Average Steps/Episode:     {final_stats['average_steps_per_episode']:.1f}")
        print(f"Total Pellets Collected:   {final_stats['total_pellets_collected']}")
        print(f"Total Times Caught:        {final_stats['total_times_caught']}")
        print(f"Total Walls Hit:           {final_stats['total_walls_hit']}")
        print(f"{'='*80}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"{'='*80}\n")
