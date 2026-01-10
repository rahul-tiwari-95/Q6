"""
Environment Wrapper with Metrics Tracking
------------------------------------------
This wrapper sits on top of our HunterGridworld environment and tracks detailed
metrics about what's happening during gameplay.

Think of this as putting sensors all over the game world to measure:
- What the agent is doing
- How the environment is responding
- What events are happening (collisions, pellet collection, etc.)
"""

import numpy as np
from typing import Tuple, Dict, Any

class EnvironmentMetricsWrapper:
    """
    Wraps the HunterGridworld environment and tracks detailed metrics.
    
    This allows us to understand exactly what's happening during training
    without cluttering the main environment code.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The HunterGridworld environment to wrap
        """
        self.env = env
        
        # Episode-level metrics
        self.episode_pellets_collected = 0
        self.episode_times_caught = 0
        self.episode_walls_hit = 0
        self.episode_total_reward = 0
        self.episode_steps = 0
        self.previous_score = 0
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and reset metrics.
        
        Returns:
            state: Initial state
            info: Additional information
        """
        # Reset metrics
        self.episode_pellets_collected = 0
        self.episode_times_caught = 0
        self.episode_walls_hit = 0
        self.episode_total_reward = 0
        self.episode_steps = 0
        self.previous_score = 0
        
        # Reset environment
        state, info = self.env.reset()
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and track metrics.
        
        Args:
            action (int): Action to take (0=Up, 1=Down, 2=Left, 3=Right)
            
        Returns:
            state: Next state
            reward: Reward received
            done: Whether episode is done
            truncated: Whether episode was truncated
            info: Information dictionary with metrics
        """
        # Get state before action (for comparison)
        state_before = self.env._get_state().copy()
        score_before = self.env.krishna.score
        lives_before = self.env.krishna.lives
        
        # Take action in environment
        state, reward, done, truncated, info = self.env.step(action)
        
        # Track metrics based on what changed
        score_after = self.env.krishna.score
        lives_after = self.env.krishna.lives
        
        # Did agent collect a pellet?
        if score_after > score_before:
            pellets_gained = (score_after - score_before) / 50  # Each pellet is 50 points
            self.episode_pellets_collected += int(pellets_gained)
        
        # Did agent get caught?
        if lives_after < lives_before:
            self.episode_times_caught += 1
        
        # Track wall hits (when reward is -5)
        if reward == -5:
            self.episode_walls_hit += 1
        
        # Track cumulative reward
        self.episode_total_reward += reward
        self.episode_steps += 1
        
        # Add metrics to info dictionary
        info['episode_pellets_collected'] = self.episode_pellets_collected
        info['episode_times_caught'] = self.episode_times_caught
        info['episode_walls_hit'] = self.episode_walls_hit
        info['episode_total_reward'] = self.episode_total_reward
        info['episode_steps'] = self.episode_steps
        info['reward_this_step'] = reward
        
        return state, reward, done, truncated, info
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics for the current episode.
        
        Returns:
            Dictionary containing episode metrics
        """
        return {
            'pellets_collected': self.episode_pellets_collected,
            'times_caught': self.episode_times_caught,
            'walls_hit': self.episode_walls_hit,
            'total_reward': self.episode_total_reward,
            'total_steps': self.episode_steps,
            'avg_reward_per_step': self.episode_total_reward / max(1, self.episode_steps)
        }
    
    def render(self):
        """Pass through render call to wrapped environment."""
        return self.env.render()
