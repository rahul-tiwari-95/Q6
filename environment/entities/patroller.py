"""
Patroller Entity Class
--------------------
Represents entities that follow predetermined patrol patterns.
Can switch between different patrol strategies.
"""

from typing import Tuple, List, Optional
import numpy as np
from .base_entity import Entity

class Patroller(Entity):
    def __init__(self, position: Tuple[int, int], grid_size: int, patrol_id: int):
        """
        Initialize Patroller entity.
        
        Args:
            position (Tuple[int, int]): Initial position
            grid_size (int): Size of the game grid
            patrol_id (int): Unique ID to determine patrol pattern
        """
        super().__init__(position, entity_id=5, grid_size=grid_size)
        self.patrol_id = patrol_id
        self.patrol_index = 0
        self.patrol_direction = 1  # 1 for clockwise, -1 for counter-clockwise
        self.steps_taken = 0
        self.stuck_counter = 0
        
    def get_patrol_pattern(self) -> List[Tuple[int, int]]:
        """
        Generate patrol pattern based on patrol_id.
        Different patterns for different patrol IDs.
        
        Returns:
            List[Tuple[int, int]]: List of relative movements for patrol
        """
        # Basic patterns: Right, Down, Left, Up
        clockwise = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        counter_clockwise = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        if self.patrol_id % 2 == 0:
            return clockwise
        return counter_clockwise
        
    def move(self, grid: np.ndarray) -> Tuple[int, int]:
        """
        Determine next move based on patrol pattern.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            Tuple[int, int]: New position after movement
        """
        pattern = self.get_patrol_pattern()
        dx, dy = pattern[self.patrol_index]
        
        # Calculate new position
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        new_pos = (new_x, new_y)
        
        # Check if move is valid
        if self._is_valid_move(new_pos, grid):
            self.steps_taken += 1
            self.stuck_counter = 0
            
            # Change direction periodically
            if self.steps_taken % 8 == 0:
                self.patrol_index = (self.patrol_index + self.patrol_direction) % 4
                
            return new_pos
        else:
            # If blocked, try turning
            self.stuck_counter += 1
            if self.stuck_counter >= 2:
                self.change_pattern()
            return self._find_alternative_move(grid)
            
    def change_pattern(self) -> None:
        """
        Change patrol pattern when stuck.
        Changes direction and potentially pattern type.
        """
        self.stuck_counter = 0
        self.patrol_direction *= -1  # Reverse direction
        self.patrol_index = (self.patrol_index + self.patrol_direction) % 4
        
    def _find_alternative_move(self, grid: np.ndarray) -> Tuple[int, int]:
        """
        Find alternative move when patrol pattern is blocked.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            Tuple[int, int]: New position
        """
        valid_moves = self.get_valid_moves(grid)
        if valid_moves:
            # Choose a move that maintains general patrol direction
            pattern = self.get_patrol_pattern()
            preferred_direction = pattern[self.patrol_index]
            
            # Score moves based on how well they match preferred direction
            def direction_score(move):
                dx = move[0] - self.position[0]
                dy = move[1] - self.position[1]
                return abs(dx - preferred_direction[0]) + abs(dy - preferred_direction[1])
            
            return min(valid_moves, key=direction_score)
        return self.position
        
    def _is_valid_move(self, pos: Tuple[int, int], grid: np.ndarray) -> bool:
        """
        Check if a move is valid.
        
        Args:
            pos (Tuple[int, int]): Position to check
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            bool: True if move is valid
        """
        x, y = pos
        return (0 <= x < self.grid_size and 
                0 <= y < self.grid_size and 
                grid[x, y] != 0)  # 0 is wall
                
    def update(self, grid: np.ndarray) -> None:
        """
        Update Patroller's state and position.
        
        Args:
            grid (np.ndarray): Current state of the game grid
        """
        new_pos = self.move(grid)
        self.update_position(new_pos)
        
        # If stuck in a loop, change pattern
        if self.is_stuck():
            self.change_pattern()
