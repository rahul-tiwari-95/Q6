"""
Greedy Bot Entity Class
----------------------
Represents entities that chase after pellets.
Uses simple pathfinding to collect pellets efficiently.
"""

from typing import Tuple, List, Set, Optional
import numpy as np
from .base_entity import Entity

class GreedyBot(Entity):
    def __init__(self, position: Tuple[int, int], grid_size: int):
        """
        Initialize Greedy Bot entity.
        
        Args:
            position (Tuple[int, int]): Initial position
            grid_size (int): Size of the game grid
        """
        super().__init__(position, entity_id=4, grid_size=grid_size)
        self.current_target = None
        self.path = []
        
    def choose_target(self, 
                     grid: np.ndarray, 
                     pellet_positions: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Choose the closest pellet as the target.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            pellet_positions (Set[Tuple[int, int]]): Set of positions containing pellets
            
        Returns:
            Optional[Tuple[int, int]]: Position of chosen pellet, or None if no pellets
        """
        if not pellet_positions:
            return None
            
        # Find closest pellet
        return min(pellet_positions,
                  key=lambda pos: self.manhattan_distance(self.position, pos))
                  
    def move(self, 
            grid: np.ndarray, 
            pellet_positions: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Determine next move towards closest pellet.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            pellet_positions (Set[Tuple[int, int]]): Set of pellet positions
            
        Returns:
            Tuple[int, int]: New position after movement
        """
        # Choose new target if needed
        if not self.current_target or self.current_target not in pellet_positions:
            self.current_target = self.choose_target(grid, pellet_positions)
            self.path = []
            
        # If no pellets left, move randomly
        if not self.current_target:
            valid_moves = self.get_valid_moves(grid)
            if valid_moves:
                return valid_moves[np.random.randint(len(valid_moves))]
            return self.position
            
        # Calculate path if needed
        if not self.path:
            self.path = self._find_path(grid, self.current_target)
            
        # Follow path or move directly towards target
        if self.path:
            next_pos = self.path.pop(0)
            if self._is_valid_move(next_pos, grid):
                return next_pos
                
        # Fallback to direct movement
        return self._direct_move(grid)
        
    def _direct_move(self, grid: np.ndarray) -> Tuple[int, int]:
        """
        Move directly towards target when pathfinding fails.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            
        Returns:
            Tuple[int, int]: New position
        """
        if not self.current_target:
            return self.position
            
        valid_moves = self.get_valid_moves(grid)
        if not valid_moves:
            return self.position
            
        # Choose move that gets us closest to target
        return min(valid_moves,
                  key=lambda pos: self.manhattan_distance(pos, self.current_target))
                  
    def _find_path(self, grid: np.ndarray, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Simple breadth-first search pathfinding.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            target (Tuple[int, int]): Target position
            
        Returns:
            List[Tuple[int, int]]: Path to target
        """
        from collections import deque
        
        frontier = deque([self.position])
        came_from = {self.position: None}
        
        while frontier:
            current = frontier.popleft()
            
            if current == target:
                break
                
            for next_pos in self.get_valid_moves(grid):
                if next_pos not in came_from:
                    frontier.append(next_pos)
                    came_from[next_pos] = current
                    
        # Reconstruct path
        path = []
        current = target
        while current != self.position:
            if current not in came_from:
                return []
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
        
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
                
    def update(self, grid: np.ndarray, pellet_positions: Set[Tuple[int, int]]) -> None:
        """
        Update Greedy Bot's state and position.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            pellet_positions (Set[Tuple[int, int]]): Set of pellet positions
        """
        new_pos = self.move(grid, pellet_positions)
        self.update_position(new_pos)
        
        # Clear path if stuck
        if self.is_stuck():
            self.path = []
            self.current_target = None
