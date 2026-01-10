"""
Kṛṣṇa Entity Class
-----------------
Represents the player character (Kṛṣṇa) in the game.
Implements player-specific behaviors and attributes.
"""

from typing import Tuple, List, Optional
import numpy as np
from .base_entity import Entity

class Krishna(Entity):
    def __init__(self, position: Tuple[int, int], grid_size: int):
        """
        Initialize Kṛṣṇa entity.
        
        Args:
            position (Tuple[int, int]): Initial position
            grid_size (int): Size of the game grid
        """
        super().__init__(position, entity_id=2, grid_size=grid_size)
        self.lives = 3
        self.score = 0
        self.pellets_collected = 0  # Track pellets for win condition
        self.invulnerable = False
        self.invulnerable_timer = 0
        
    def move(self, action: int, grid: np.ndarray) -> Tuple[int, int]:
        """
        Move Kṛṣṇa based on the action.
        
        Args:
            action (int): 0=Up, 1=Down, 2=Left, 3=Right
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            Tuple[int, int]: New position after movement
        """
        x, y = self.position
        new_pos = self.position
        
        # Calculate new position based on action
        if action == 0 and x > 0:  # Up
            new_pos = (x-1, y)
        elif action == 1 and x < self.grid_size-1:  # Down
            new_pos = (x+1, y)
        elif action == 2 and y > 0:  # Left
            new_pos = (x, y-1)
        elif action == 3 and y < self.grid_size-1:  # Right
            new_pos = (x, y+1)
            
        # Check if new position is valid (not a wall)
        if grid[new_pos] != 0:  # 0 is wall
            self.update_position(new_pos)
            
        return self.position
        
    def collect_pellet(self) -> None:
        """
        Handle pellet collection logic.
        Increases score and pellet count.
        """
        self.score += 50
        self.pellets_collected += 1
        
    def take_damage(self) -> bool:
        """
        Handle damage taken from enemies.
        
        Returns:
            bool: True if Kṛṣṇa is still alive
        """
        if not self.invulnerable:
            self.lives -= 1
            if self.lives > 0:
                self.invulnerable = True
                self.invulnerable_timer = 30  # Invulnerable for 30 frames
        return self.lives > 0
        
    def update(self) -> None:
        """
        Update Kṛṣṇa's state (called every frame).
        Handles things like invulnerability timer.
        """
        if self.invulnerable:
            self.invulnerable_timer -= 1
            if self.invulnerable_timer <= 0:
                self.invulnerable = False
                
    def get_state_info(self) -> dict:
        """
        Get current state information for the entity.
        
        Returns:
            dict: Current state including position, lives, score, etc.
        """
        return {
            'position': self.position,
            'lives': self.lives,
            'score': self.score,
            'pellets_collected': self.pellets_collected,
            'invulnerable': self.invulnerable,
            'invulnerable_timer': self.invulnerable_timer
        }
