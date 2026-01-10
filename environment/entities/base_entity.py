"""
Base Entity Class
----------------
Defines the common attributes and methods for all entities in the game.
This serves as a foundation for specific entity types like Kṛṣṇa, Hunter, etc.
"""

from typing import Tuple, List, Optional
import numpy as np

class Entity:
    def __init__(self, position: Tuple[int, int], entity_id: int, grid_size: int):
        """
        Initialize base entity.
        
        Args:
            position (Tuple[int, int]): Initial (x, y) position
            entity_id (int): Unique identifier for this entity type
            grid_size (int): Size of the game grid
        """
        self.position = position
        self.entity_id = entity_id
        self.grid_size = grid_size
        self.previous_positions = []  # Track last few positions for pattern analysis
        self.max_history = 5  # Maximum number of previous positions to store

    def update_position(self, new_position: Tuple[int, int]) -> None:
        """
        Update entity position and maintain position history.
        
        Args:
            new_position (Tuple[int, int]): New (x, y) position
        """
        self.previous_positions.append(self.position)
        if len(self.previous_positions) > self.max_history:
            self.previous_positions.pop(0)
        self.position = new_position

    def get_valid_moves(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all valid moves from current position.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            List[Tuple[int, int]]: List of valid (x, y) positions
        """
        x, y = self.position
        possible_moves = [
            (x-1, y),  # Up
            (x+1, y),  # Down
            (x, y-1),  # Left
            (x, y+1)   # Right
        ]
        
        # Filter out invalid moves (walls or out of bounds)
        valid_moves = []
        for move in possible_moves:
            new_x, new_y = move
            if (0 <= new_x < self.grid_size and 
                0 <= new_y < self.grid_size and 
                grid[new_x, new_y] != 0):  # 0 is wall
                valid_moves.append(move)
                
        return valid_moves

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1 (Tuple[int, int]): First position
            pos2 (Tuple[int, int]): Second position
            
        Returns:
            int: Manhattan distance between positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_stuck(self) -> bool:
        """
        Check if entity is stuck in a loop.
        
        Returns:
            bool: True if entity is repeating positions
        """
        if len(self.previous_positions) < self.max_history:
            return False
        # Check if we're oscillating between positions
        return len(set(self.previous_positions)) <= 2
