"""
Hunter Entity Class
-----------------
Represents the main antagonist that actively pursues Kṛṣṇa.
Implements intelligent chase behavior with path prediction.
"""

from typing import Tuple, List, Optional
import numpy as np
from .base_entity import Entity

class Hunter(Entity):
    def __init__(self, position: Tuple[int, int], grid_size: int):
        """
        Initialize Hunter entity.
        
        Args:
            position (Tuple[int, int]): Initial position
            grid_size (int): Size of the game grid
        """
        super().__init__(position, entity_id=3, grid_size=grid_size)
        self.target_position = None
        self.path = []
        self.prediction_depth = 3  # How many steps ahead to predict
        
    def predict_target_position(self, 
                              target_pos: Tuple[int, int], 
                              target_previous_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Predict where the target (Kṛṣṇa) will be in the next few steps.
        Uses simple velocity calculation from previous positions.
        
        Args:
            target_pos (Tuple[int, int]): Current target position
            target_previous_positions (List[Tuple[int, int]]): Target's movement history
            
        Returns:
            Tuple[int, int]: Predicted future position
        """
        if len(target_previous_positions) < 2:
            return target_pos
            
        # Calculate average velocity
        recent_positions = target_previous_positions[-2:]
        dx = sum(p[0] - q[0] for p, q in zip(recent_positions[1:], recent_positions[:-1]))
        dy = sum(p[1] - q[1] for p, q in zip(recent_positions[1:], recent_positions[:-1]))
        
        # Predict future position
        predicted_x = target_pos[0] + (dx * self.prediction_depth)
        predicted_y = target_pos[1] + (dy * self.prediction_depth)
        
        # Ensure predicted position is within bounds
        predicted_x = max(0, min(predicted_x, self.grid_size - 1))
        predicted_y = max(0, min(predicted_y, self.grid_size - 1))
        
        return (int(predicted_x), int(predicted_y))
        
    def choose_move(self, 
                   grid: np.ndarray, 
                   target_pos: Tuple[int, int],
                   target_previous_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Determine the next move to chase the target.
        Uses A* pathfinding if path is empty, otherwise follows existing path.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            target_pos (Tuple[int, int]): Current position of the target
            target_previous_positions (List[Tuple[int, int]]): Target's movement history
            
        Returns:
            Tuple[int, int]: Next position to move to
        """
        # Predict where target is heading
        predicted_pos = self.predict_target_position(target_pos, target_previous_positions)
        
        # If we need to calculate a new path
        if not self.path or self.target_position != predicted_pos:
            self.target_position = predicted_pos
            self.path = self._find_path(grid, self.target_position)
        
        # Get next move from path
        if self.path:
            next_pos = self.path.pop(0)
            if self._is_valid_move(next_pos, grid):
                return next_pos
                
        # If path is blocked, use simple chase logic
        return self._chase_move(grid, target_pos)
        
    def _chase_move(self, grid: np.ndarray, target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Simple chase behavior when pathfinding fails.
        Moves toward target using manhattan distance.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            target_pos (Tuple[int, int]): Position of the target
            
        Returns:
            Tuple[int, int]: Next position to move to
        """
        valid_moves = self.get_valid_moves(grid)
        if not valid_moves:
            return self.position
            
        # Choose move that minimizes distance to target
        return min(valid_moves, 
                  key=lambda pos: self.manhattan_distance(pos, target_pos))
                  
    def _find_path(self, grid: np.ndarray, target_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding implementation.
        Finds optimal path to target considering walls.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            target_pos (Tuple[int, int]): Target position
            
        Returns:
            List[Tuple[int, int]]: List of positions forming a path
        """
        from queue import PriorityQueue
        
        frontier = PriorityQueue()
        frontier.put((0, self.position))
        came_from = {self.position: None}
        cost_so_far = {self.position: 0}
        
        while not frontier.empty():
            current = frontier.get()[1]
            
            if current == target_pos:
                break
                
            for next_pos in self.get_valid_moves(grid):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.manhattan_distance(next_pos, target_pos)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current
                    
        # Reconstruct path
        path = []
        current = target_pos
        while current != self.position:
            if current not in came_from:
                return []  # No path found
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
        
    def _is_valid_move(self, pos: Tuple[int, int], grid: np.ndarray) -> bool:
        """
        Check if a move is valid (within bounds and not a wall).
        
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
                
    def update(self, 
              grid: np.ndarray, 
              target_pos: Tuple[int, int],
              target_previous_positions: List[Tuple[int, int]]) -> None:
        """
        Update Hunter's state and position.
        
        Args:
            grid (np.ndarray): Current state of the game grid
            target_pos (Tuple[int, int]): Current position of the target
            target_previous_positions (List[Tuple[int, int]]): Target's movement history
        """
        new_pos = self.choose_move(grid, target_pos, target_previous_positions)
        self.update_position(new_pos)
        
        # Clear path if we're stuck
        if self.is_stuck():
            self.path = []
