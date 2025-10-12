"""
Project Q6: Sui Generis - Main Training Loop
Author: Rahul Tiwari
"""

import gymnasium as gym
import numpy as np
import torch
from agent.dq_agent import DQNAgent
from environment.hunter_gridworld import HunterGridworld

def main():
    # Initialize environment and agent
    env = HunterGridworld()
    agent = DQNAgent(state_size=625, action_size=4)
    
    # Training parameters
    n_episodes = 10000
    max_steps = 1000
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            step += 1
            
        print(f"Episode: {episode + 1}/{n_episodes}, Score: {total_reward}")

if __name__ == "__main__":
    main()
