"""
Progressive Hunter Difficulty System
=====================================
Version: 0.5
Date: December 12, 2025

Implements 4 levels of Hunter intelligence for curriculum learning.

PHILOSOPHY:
-----------
Instead of introducing the Hunter late in training (causing a difficulty shock),
we introduce it from episode 1 but gradually increase its intelligence.

This is like learning to box:
- Don't spar with beginners for months, then fight Mike Tyson
- Spar with beginners, then intermediates, then pros, then champions
- Build Hunter-specific skills from day 1, but progressively

PROGRESSIVE DIFFICULTY LEVELS:
-------------------------------
Phase 1: Random Hunter      - Moves randomly, catches Krishna by luck only
Phase 2: Greedy Hunter      - Moves toward Krishna, but can be trapped by walls
Phase 3: Smart Greedy       - Like Greedy but with wall avoidance
Phase 4: A* Hunter          - Optimal pathfinding, cannot be trapped

This gradual progression allows the agent to:
- Learn Hunter avoidance as a core skill from episode 1
- Build confidence before facing optimal adversary
- Develop counter-strategies at each difficulty level
- Transfer learning across phases (no catastrophic forgetting)
"""

import random
from typing import Tuple, List, Optional
from collections import deque


class BaseProgressiveHunter:
    """
    Base class for all progressive hunters
    
    Provides common functionality:
    - Position management
    - Valid move calculation
    - Movement execution
    """
    
    def __init__(self, pos: Tuple[int, int], grid_size: int = 25):
        """
        Initialize hunter
        
        Args:
            pos: Starting position (x, y)
            grid_size: Grid dimension (default 25x25)
        """
        self.pos = list(pos)
        self.grid_size = grid_size
        self.speed = 1
        self.name = "Base Hunter"
        self.difficulty_level = 0
        self.color = (255, 0, 0)  # Red
    
    @property
    def position(self) -> Tuple[int, int]:
        """
        Compatibility property for position access.
        Returns tuple version of self.pos for compatibility with original Hunter API.
        """
        return tuple(self.pos)
    
    def get_valid_moves(self, grid) -> List[str]:
        """
        Get list of valid moves (not into walls)
        
        Args:
            grid: 2D array representing the environment
        
        Returns:
            List of valid direction strings
        """
        valid = []
        x, y = self.pos
        
        moves = {
            'UP': (x, y - 1),
            'DOWN': (x, y + 1),
            'LEFT': (x - 1, y),
            'RIGHT': (x + 1, y)
        }
        
        for direction, (nx, ny) in moves.items():
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if grid[ny, nx] != 0:  # Not a wall (0 = WALL, not 1 = PELLET)
                    valid.append(direction)
        
        return valid if valid else ['UP']  # Fallback if somehow no valid moves
    
    def move(self, direction: str):
        """
        Execute movement in specified direction
        
        Args:
            direction: One of 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        if direction == 'UP':
            self.pos[1] = max(0, self.pos[1] - 1)
        elif direction == 'DOWN':
            self.pos[1] = min(self.grid_size - 1, self.pos[1] + 1)
        elif direction == 'LEFT':
            self.pos[0] = max(0, self.pos[0] - 1)
        elif direction == 'RIGHT':
            self.pos[0] = min(self.grid_size - 1, self.pos[0] + 1)
    
    def get_state_info(self) -> dict:
        """Get hunter state information"""
        return {
            'pos': tuple(self.pos),
            'name': self.name,
            'difficulty': self.difficulty_level
        }
    
    def update(self, grid, krishna_pos: Tuple[int, int], krishna_previous_positions: List[Tuple[int, int]] = None):
        """
        Compatibility method for original Hunter API.
        
        This method is called by HunterGridworld to update the Hunter's position.
        It internally uses the progressive Hunter's choose_action() and move() methods.
        
        Args:
            grid: 2D array representing the environment
            krishna_pos: Krishna's current position (x, y)
            krishna_previous_positions: Krishna's movement history (unused by progressive Hunters)
        """
        # Choose action based on this Hunter's specific algorithm
        action = self.choose_action(krishna_pos, grid)
        
        # Execute the move
        self.move(action)


class RandomHunter(BaseProgressiveHunter):
    """
    PHASE 1: Random Hunter
    ======================
    
    BEHAVIOR:
    ---------
    Moves completely randomly, ignoring Krishna's position.
    Occasionally catches Krishna by luck when their paths cross.
    
    LEARNING GOALS FOR AGENT:
    --------------------------
    - Learn: "There's a red thing that sometimes catches me"
    - Focus: Navigation fundamentals and pellet collection
    - Pressure: Very low - Hunter is essentially another wandering bot
    
    DIFFICULTY: ★☆☆☆ (Very Low)
    
    EXPECTED AGENT PERFORMANCE:
    ---------------------------
    - Win rate: 60-70%
    - Pellets/episode: 3.0-3.5
    - Catches/episode: 0.2-0.5
    """
    
    def __init__(self, pos: Tuple[int, int], grid_size: int = 25):
        super().__init__(pos, grid_size)
        self.name = "Random Hunter"
        self.difficulty_level = 1
    
    def choose_action(self, krishna_pos: Tuple[int, int], grid) -> str:
        """
        Choose completely random movement
        
        Args:
            krishna_pos: Krishna's position (ignored)
            grid: Environment grid
        
        Returns:
            Random valid direction
        """
        valid_moves = self.get_valid_moves(grid)
        return random.choice(valid_moves)


class GreedyHunter(BaseProgressiveHunter):
    """
    PHASE 2: Greedy Hunter
    ======================
    
    BEHAVIOR:
    ---------
    Always moves in the direction that reduces Manhattan distance to Krishna.
    Takes direct path, no pathfinding - can be trapped behind walls.
    
    ALGORITHM:
    ----------
    1. Calculate dx = krishna_x - hunter_x
    2. Calculate dy = krishna_y - hunter_y
    3. If |dx| > |dy|: move horizontally toward Krishna
    4. Else: move vertically toward Krishna
    5. If chosen direction is blocked: move randomly
    
    LEARNING GOALS FOR AGENT:
    --------------------------
    - Learn: "Hunter follows me, but I can use walls and distance"
    - Focus: Basic evasion tactics emerge
    - Pressure: Medium - Must start thinking about Hunter position
    - Exploitable: Agent can trap Hunter behind walls
    
    DIFFICULTY: ★★☆☆ (Low-Medium)
    
    EXPECTED AGENT PERFORMANCE:
    ---------------------------
    - Win rate: 50-65%
    - Pellets/episode: 2.5-3.2
    - Catches/episode: 0.4-0.8
    """
    
    def __init__(self, pos: Tuple[int, int], grid_size: int = 25):
        super().__init__(pos, grid_size)
        self.name = "Greedy Hunter"
        self.difficulty_level = 2
    
    def choose_action(self, krishna_pos: Tuple[int, int], grid) -> str:
        """
        Move in direction that reduces Manhattan distance
        
        Args:
            krishna_pos: Krishna's position
            grid: Environment grid
        
        Returns:
            Direction that reduces distance
        """
        dx = krishna_pos[0] - self.pos[0]
        dy = krishna_pos[1] - self.pos[1]
        
        # Prefer horizontal or vertical based on larger distance
        if abs(dx) > abs(dy):
            preferred = 'RIGHT' if dx > 0 else 'LEFT'
        else:
            preferred = 'DOWN' if dy > 0 else 'UP'
        
        valid_moves = self.get_valid_moves(grid)
        
        # Try preferred direction first
        if preferred in valid_moves:
            return preferred
        else:
            # Fallback to any valid move if preferred is blocked
            return random.choice(valid_moves)


class SmartGreedyHunter(BaseProgressiveHunter):
    """
    PHASE 3: Smart Greedy Hunter
    ============================
    
    BEHAVIOR:
    ---------
    Like GreedyHunter but with intelligent fallback strategies.
    - Primary: Move in direction of larger distance component
    - Secondary: Move in direction of smaller distance component
    - Tertiary: Move perpendicular to avoid getting stuck
    - Still no pathfinding, but doesn't trap itself easily
    
    ALGORITHM:
    ----------
    1. Calculate primary direction (larger dx/dy)
    2. Calculate secondary direction (smaller dx/dy)
    3. Try primary → secondary → perpendicular → any valid
    4. Adaptively chooses best available option
    
    LEARNING GOALS FOR AGENT:
    --------------------------
    - Learn: "Hunter is more persistent, can't trap it easily"
    - Focus: Strategic positioning and timing
    - Pressure: High - Must actively evade, not just hide
    - Still beatable: No pathfinding = Krishna can outmaneuver
    
    DIFFICULTY: ★★★☆ (Medium-High)
    
    EXPECTED AGENT PERFORMANCE:
    ---------------------------
    - Win rate: 40-55%
    - Pellets/episode: 2.0-2.8
    - Catches/episode: 0.8-1.2
    """
    
    def __init__(self, pos: Tuple[int, int], grid_size: int = 25):
        super().__init__(pos, grid_size)
        self.name = "Smart Greedy Hunter"
        self.difficulty_level = 3
    
    def choose_action(self, krishna_pos: Tuple[int, int], grid) -> str:
        """
        Move toward Krishna with smart fallback strategy
        
        Args:
            krishna_pos: Krishna's position
            grid: Environment grid
        
        Returns:
            Best available direction
        """
        dx = krishna_pos[0] - self.pos[0]
        dy = krishna_pos[1] - self.pos[1]
        
        # Primary direction (larger distance)
        if abs(dx) > abs(dy):
            primary = 'RIGHT' if dx > 0 else 'LEFT'
            secondary = 'DOWN' if dy > 0 else 'UP'
        else:
            primary = 'DOWN' if dy > 0 else 'UP'
            secondary = 'RIGHT' if dx > 0 else 'LEFT'
        
        valid_moves = self.get_valid_moves(grid)
        
        # Try primary direction
        if primary in valid_moves:
            return primary
        
        # Try secondary direction
        if secondary in valid_moves:
            return secondary
        
        # Try perpendicular moves (avoid getting stuck in corners)
        perpendicular = {
            'UP': ['LEFT', 'RIGHT'],
            'DOWN': ['LEFT', 'RIGHT'],
            'LEFT': ['UP', 'DOWN'],
            'RIGHT': ['UP', 'DOWN']
        }
        
        for move in perpendicular.get(primary, []):
            if move in valid_moves:
                return move
        
        # Last resort: any valid move
        return random.choice(valid_moves)


class AStarHunter(BaseProgressiveHunter):
    """
    PHASE 4: A* Hunter (Optimal Pathfinding)
    =========================================
    
    BEHAVIOR:
    ---------
    Uses A* algorithm to find optimal path to Krishna.
    - Always takes shortest possible path
    - Cannot be trapped or confused by walls
    - Relentless pursuit with perfect navigation
    
    ALGORITHM:
    ----------
    A* Search with Manhattan distance heuristic:
    1. Open set = priority queue sorted by f(n) = g(n) + h(n)
    2. g(n) = actual distance from start
    3. h(n) = Manhattan distance to goal (heuristic)
    4. Explores minimum cost path first
    5. Returns optimal path when goal reached
    
    LEARNING GOALS FOR AGENT:
    --------------------------
    - Learn: "Hunter is relentless, I need advanced strategies"
    - Focus: Pattern emergence, advanced evasion, efficiency
    - Pressure: Maximum - Must use everything learned in Phases 1-3
    - No exploits: Hunter takes optimal path always
    
    DIFFICULTY: ★★★★ (Expert)
    
    EXPECTED AGENT PERFORMANCE:
    ---------------------------
    - Win rate: 25-40% (challenging but achievable)
    - Pellets/episode: 1.5-2.5
    - Catches/episode: 1.5-2.5
    
    PERFORMANCE NOTES:
    ------------------
    This is the final boss. Agent should demonstrate:
    - Strategic pellet collection (prioritize safe pellets)
    - Risk management (know when to retreat)
    - Pattern recognition (predict Hunter movement)
    - Efficient navigation (no wasted moves)
    """
    
    def __init__(self, pos: Tuple[int, int], grid_size: int = 25):
        super().__init__(pos, grid_size)
        self.name = "A* Hunter"
        self.difficulty_level = 4
    
    def choose_action(self, krishna_pos: Tuple[int, int], grid) -> str:
        """
        Use A* pathfinding to find optimal move
        
        Args:
            krishna_pos: Krishna's position (target)
            grid: Environment grid
        
        Returns:
            Direction toward optimal path
        """
        path = self.a_star_search(tuple(self.pos), tuple(krishna_pos), grid)
        
        if len(path) > 1:
            next_pos = path[1]
            dx = next_pos[0] - self.pos[0]
            dy = next_pos[1] - self.pos[1]
            
            if dx > 0:
                return 'RIGHT'
            elif dx < 0:
                return 'LEFT'
            elif dy > 0:
                return 'DOWN'
            elif dy < 0:
                return 'UP'
        
        # Fallback to greedy if A* somehow fails
        return GreedyHunter.choose_action(self, krishna_pos, grid)
    
    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int], grid) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm
        
        Args:
            start: Starting position
            goal: Target position
            grid: Environment grid
        
        Returns:
            List of positions from start to goal (optimal path)
        """
        from heapq import heappush, heappop
        
        def heuristic(pos):
            """Manhattan distance heuristic"""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Priority queue: (f_score, position)
        open_set = []
        heappush(open_set, (0, start))
        
        # Track path
        came_from = {}
        
        # g_score: actual cost from start
        g_score = {start: 0}
        
        # f_score: g_score + heuristic
        f_score = {start: heuristic(start)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            # Goal reached - reconstruct path
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            # Explore neighbors
            x, y = current
            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1)
            ]
            
            for nx, ny in neighbors:
                # Validate neighbor
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if grid[ny, nx] != 0:  # Not a wall (0 = WALL, not 1 = PELLET)
                        neighbor = (nx, ny)
                        tentative_g = g_score[current] + 1
                        
                        # Found better path to neighbor
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + heuristic(neighbor)
                            
                            # Add to open set if not already there
                            if neighbor not in [item[1] for item in open_set]:
                                heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found - return start position
        return [start]


def create_hunter_for_phase(phase: int, pos: Tuple[int, int], grid_size: int = 25) -> BaseProgressiveHunter:
    """
    Factory function: Create appropriate Hunter for curriculum phase
    
    Args:
        phase: Curriculum phase (1-4)
        pos: Starting position
        grid_size: Grid dimension
    
    Returns:
        Hunter instance appropriate for phase
    
    Example:
        >>> hunter = create_hunter_for_phase(1, (10, 10))
        >>> print(hunter.name)
        'Random Hunter'
        >>> print(hunter.difficulty_level)
        1
    """
    hunter_classes = {
        1: RandomHunter,
        2: GreedyHunter,
        3: SmartGreedyHunter,
        4: AStarHunter
    }
    
    hunter_class = hunter_classes.get(phase, AStarHunter)
    return hunter_class(pos, grid_size)
