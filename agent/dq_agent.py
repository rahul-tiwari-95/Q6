"""
DQN Agent Implementation
----------------------
Implements the Deep Q-Network agent with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
import csv
import os
from typing import Tuple, List, Dict, Any
from model.q_network import QNetwork

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experiences.
    Implements efficient storage and random sampling for experience replay.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, 
            action: int, 
            reward: float, 
            next_state: np.ndarray, 
            done: bool) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # Transpose the batch
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8))
                
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent implementation.
    Combines Q-Network with experience replay and target network for stable learning.
    """
    
    def __init__(self,
                 state_size: int = 625,
                 action_size: int = 4,
                 hidden_sizes: Tuple[int, ...] = (256, 128),
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 1e-3,
                 batch_size: int = 64,
                 buffer_size: int = 100000,
                 update_every: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize DQN Agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            hidden_sizes (tuple): Sizes of hidden layers
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            tau (float): Soft update parameter
            batch_size (int): Mini-batch size
            buffer_size (int): Replay buffer size
            update_every (int): How often to update network
            device (str): Device to run on (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.device = device
        
        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, hidden_sizes, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, hidden_sizes, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
        # Exploration parameters (OPTIMIZED v0.5 - PHASE-ADAPTIVE EPSILON)
        # epsilon: Probability of random action (exploration vs exploitation)
        # Start high (1.0 = 100% random) for initial exploration
        # Decay slower to maintain exploration throughout training
        # RESET at phase transitions to explore new Hunter difficulties
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 5% minimum exploration maintained
        self.epsilon_decay = 0.9999  # Decay rate within phases
        # With 0.9999: 
        #   - Reaches 0.5 at ~6,900 episodes
        #   - Reaches 0.1 at ~460 episodes
        #   - Reaches 0.05 at ~3,000 episodes
        
        # PHASE-SPECIFIC EPSILON RESET VALUES (v0.5 NEW)
        # When transitioning to new phase, reset epsilon to these values:
        self.phase_epsilon_reset = {
            1: 1.0,   # Phase 1: Full exploration (Random Hunter)
            2: 0.5,   # Phase 2: Balanced (Greedy Hunter)
            3: 0.3,   # Phase 3: Mostly exploit (Smart Greedy)
            4: 0.2    # Phase 4: Strategic exploration (A* Hunter)
        }
        
        # Q-VALUE LOGGING: Track local vs target network performance
        self.q_value_log = []  # Stores Q-value comparison data
        self.log_q_values = False  # Enable/disable Q-value logging
        self.q_log_file = None  # CSV file for Q-value logs
        self.learning_step = 0  # Track learning iterations
        
    def step(self, state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool,
             episode: int = 0) -> None:
        """
        Store experience in replay memory and learn if it's time.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            episode: Current episode number (for logging)
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn(current_episode=episode)
            
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether we're training (use epsilon-greedy) or not
            
        Returns:
            int: Chosen action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def get_q_values_comparison(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Get Q-values from both local and target networks for comparison.
        This helps understand how the networks are learning and converging.
        
        Args:
            state: Current state
            
        Returns:
            dict: Contains local Q-values, target Q-values, differences, and metrics
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        
        with torch.no_grad():
            local_q_values = self.qnetwork_local(state_tensor).cpu().numpy()[0]
            target_q_values = self.qnetwork_target(state_tensor).cpu().numpy()[0]
        
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        
        # Calculate metrics
        q_diff = local_q_values - target_q_values
        agreement = np.argmax(local_q_values) == np.argmax(target_q_values)
        max_q_diff = np.abs(np.max(local_q_values) - np.max(target_q_values))
        mean_q_diff = np.mean(np.abs(q_diff))
        
        return {
            'local_q_values': local_q_values,
            'target_q_values': target_q_values,
            'local_best_action': np.argmax(local_q_values),
            'target_best_action': np.argmax(target_q_values),
            'q_differences': q_diff,
            'agreement': agreement,
            'max_q_diff': max_q_diff,
            'mean_q_diff': mean_q_diff,
            'local_max_q': np.max(local_q_values),
            'target_max_q': np.max(target_q_values)
        }
    
    def enable_q_logging(self, log_dir: str) -> None:
        """
        Enable Q-value logging to CSV file.
        
        Args:
            log_dir: Directory to save the Q-value log file
        """
        self.log_q_values = True
        log_path = os.path.join(log_dir, 'q_values_comparison.csv')
        self.q_log_file = open(log_path, 'w', newline='')
        
        fieldnames = [
            'learning_step', 'episode', 'local_best_action', 'target_best_action',
            'agreement', 'max_q_diff', 'mean_q_diff', 'local_max_q', 'target_max_q',
            'local_q0', 'local_q1', 'local_q2', 'local_q3',
            'target_q0', 'target_q1', 'target_q2', 'target_q3'
        ]
        
        self.q_csv_writer = csv.DictWriter(self.q_log_file, fieldnames=fieldnames)
        self.q_csv_writer.writeheader()
        self.q_log_file.flush()
        
        print(f"✅ Q-value logging enabled: {log_path}")
    
    def disable_q_logging(self) -> None:
        """Disable Q-value logging and close file."""
        self.log_q_values = False
        if self.q_log_file:
            self.q_log_file.close()
            self.q_log_file = None
        
    def learn(self, current_episode: int = 0) -> Dict[str, float]:
        """
        Update value parameters using batch of experience tuples.
        Implements DQN learning algorithm with target network.
        
        Args:
            current_episode: Current episode number (for logging)
            
        Returns:
            dict: Learning metrics including loss and Q-value statistics
        """
        self.learning_step += 1
        
        # Sample random batch from memory
        states, actions, rewards, next_states, dones = map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        )
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1).long())
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update()
        
        # NOTE: Epsilon decay is now handled per-episode in main.py, not per-learning-step
        # This prevents epsilon from decaying too quickly (was decaying every 4 steps!)
        
        # === Q-VALUE LOGGING ===
        # Log Q-value comparison every 100 learning steps to avoid excessive overhead
        if self.log_q_values and self.learning_step % 100 == 0:
            # Use first state from batch for Q-value comparison
            sample_state = states[0].cpu().numpy()
            q_comparison = self.get_q_values_comparison(sample_state)
            
            # Write to CSV
            log_entry = {
                'learning_step': self.learning_step,
                'episode': current_episode,
                'local_best_action': int(q_comparison['local_best_action']),
                'target_best_action': int(q_comparison['target_best_action']),
                'agreement': int(q_comparison['agreement']),
                'max_q_diff': float(q_comparison['max_q_diff']),
                'mean_q_diff': float(q_comparison['mean_q_diff']),
                'local_max_q': float(q_comparison['local_max_q']),
                'target_max_q': float(q_comparison['target_max_q']),
                'local_q0': float(q_comparison['local_q_values'][0]),
                'local_q1': float(q_comparison['local_q_values'][1]),
                'local_q2': float(q_comparison['local_q_values'][2]),
                'local_q3': float(q_comparison['local_q_values'][3]),
                'target_q0': float(q_comparison['target_q_values'][0]),
                'target_q1': float(q_comparison['target_q_values'][1]),
                'target_q2': float(q_comparison['target_q_values'][2]),
                'target_q3': float(q_comparison['target_q_values'][3])
            }
            
            self.q_csv_writer.writerow(log_entry)
            if self.learning_step % 1000 == 0:  # Flush every 1000 steps
                self.q_log_file.flush()
        
        # Return learning metrics
        return {
            'loss': loss.item(),
            'mean_q_expected': Q_expected.mean().item(),
            'mean_q_target': Q_targets.mean().item()
        }
        
    def soft_update(self) -> None:
        """
        Soft update model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(),
            self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def reset_epsilon_for_phase(self, phase: int) -> Tuple[float, float]:
        """
        Reset epsilon when transitioning to new curriculum phase (v0.5 NEW)
        
        This allows fresh exploration when Hunter difficulty increases.
        Like learning to swim progressively deeper - need to re-explore at each depth.
        
        Args:
            phase: New curriculum phase (1-4)
        
        Returns:
            Tuple of (old_epsilon, new_epsilon)
        
        Example:
            >>> agent.epsilon = 0.08  # Almost at minimum from Phase 3
            >>> old, new = agent.reset_epsilon_for_phase(4)  # Phase 4 transition
            >>> print(f"Epsilon reset: {old:.4f} → {new:.4f}")
            Epsilon reset: 0.0800 → 0.2000
        """
        old_epsilon = self.epsilon
        new_epsilon = self.phase_epsilon_reset.get(phase, self.epsilon)
        
        # Only reset if new value is higher (don't decrease epsilon)
        if new_epsilon > self.epsilon:
            self.epsilon = max(new_epsilon, self.epsilon_min)
            return old_epsilon, self.epsilon
        
        return old_epsilon, old_epsilon  # No change
    
    def decay_epsilon(self) -> float:
        """
        Decay epsilon by the decay rate. Should be called once per episode.
        
        Returns:
            float: New epsilon value after decay
            
        Note:
            This method should be called at the END of each episode in the training loop,
            not during learning steps, to ensure proper exploration decay rate.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon
            
    def save(self, path: str) -> None:
        """
        Save agent's state.
        
        Args:
            path (str): Path to save the agent
        """
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str) -> None:
        """
        Load agent's state.
        
        Args:
            path (str): Path to load the agent from
        """
        checkpoint = torch.load(path)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
