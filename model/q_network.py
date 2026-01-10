"""
Neural Network Model
------------------
Defines the Multi-Layer Perceptron (MLP) architecture for the DQN agent.
This network maps the state space to Q-values for each possible action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    Architecture: 625 -> 256 -> 128 -> 4
    Uses ReLU activation and includes optional dropout for regularization.
    """
    
    def __init__(self, 
                 input_size: int = 625, 
                 hidden_sizes: Tuple[int, ...] = (256, 128),
                 output_size: int = 4,
                 dropout_rate: float = 0.1):
        """
        Initialize the Q-Network.
        
        Args:
            input_size (int): Size of input (flattened grid state)
            hidden_sizes (tuple): Sizes of hidden layers
            output_size (int): Number of actions
            dropout_rate (float): Dropout probability for regularization
        """
        super(QNetwork, self).__init__()
        
        # Create list of layer sizes
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        
        # Create layers dynamically based on specified sizes
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize weights using He initialization.
        This helps prevent vanishing/exploding gradients.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Ensure input is float tensor
        x = x.float()
        
        # Pass through all layers except last
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        # Final layer (no activation or dropout)
        x = self.layers[-1](x)
        
        return x
    
    def save(self, path: str) -> None:
        """
        Save the model weights to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        """
        Load the model weights from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.load_state_dict(torch.load(path))
