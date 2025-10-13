"""
HunterGridworld Environment
--------------------------
This is a custom Gymnasium environment that implements a 25x25 grid world where
Kṛṣṇa (our learning agent) must collect pellets while avoiding enemies.

Environment Details:
- Grid Size: 25x25
- State Space: 625 elements (flattened grid)
- Action Space: 4 discrete actions (Up, Down, Left, Right)
- Entities: 
    * Kṛṣṇa (player) = 2
    * Hunter (enemy) = 3
    * Greedy Bots = 4
    * Patrollers = 5
    * Walls = 0
    * Pellets = 1
    * Empty Space = 6
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class HunterGridworld(gym.Env):
    """
    Custom Environment that follows gym interface.
    This represents our game world where Kṛṣṇa must navigate and survive.
    """
    # Class constants for better code readability
    WALL = 0
    PELLET = 1
    KRISHNA = 2
    HUNTER = 3
    GREEDY = 4
    PATROL = 5
    EMPTY = 6

    def __init__(self):
        super(HunterGridworld, self).__init__()

        # Grid dimensions
        self.grid_size = 25
        self.state_size = self.grid_size * self.grid_size

        # Define action and observation spaces
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 625 cells (25x25 grid)
        # Each cell can contain one of 7 possible values (0-6)
        self.observation_space = spaces.Box(
            low=0, 
            high=6,
            shape=(self.state_size,), 
            dtype=np.int32
        )

        # Game state variables
        self.grid = None
        self.krishna_pos = None
        self.hunter_pos = None
        self.greedy_positions = []
        self.patrol_positions = []
        self.pellet_positions = set()
        self.score = 0
        self.lives = 3

    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        Returns:
            state: Initial state of the environment
            info: Additional information (empty dict in this case)
        """
        super().reset(seed=seed)
        
        # Initialize empty grid
        self.grid = np.full((self.grid_size, self.grid_size), self.EMPTY)
        
        # Generate walls (approximately 20% of the grid)
        self._generate_walls()
        
        # Place pellets (approximately 10% of remaining space)
        self._place_pellets()
        
        # Place Kṛṣṇa in a random empty cell
        self.krishna_pos = self._get_random_empty_position()
        self.grid[self.krishna_pos] = self.KRISHNA
        
        # Place Hunter
        self.hunter_pos = self._get_random_empty_position()
        self.grid[self.hunter_pos] = self.HUNTER
        
        # Place 2 Greedy Bots
        self.greedy_positions = []
        for _ in range(2):
            pos = self._get_random_empty_position()
            self.greedy_positions.append(pos)
            self.grid[pos] = self.GREEDY
            
        # Place 2 Patrollers
        self.patrol_positions = []
        for _ in range(2):
            pos = self._get_random_empty_position()
            self.patrol_positions.append(pos)
            self.grid[pos] = self.PATROL
        
        # Reset score and lives
        self.score = 0
        self.lives = 3
        
        return self._get_state(), {}

    def step(self, action):
        """
        Execute one time step within the environment.
        Args:
            action: int (0=Up, 1=Down, 2=Left, 3=Right)
        Returns:
            state: Current state of environment
            reward: Reward from the action
            done: Whether the episode has ended
            truncated: Whether episode was artificially terminated
            info: Additional information
        """
        # Initialize reward for this step
        reward = -0.1  # Small penalty for each step to encourage efficiency
        done = False
        
        # Get new position based on action
        new_pos = self._get_new_position(self.krishna_pos, action)
        
        # Check if the move is valid (not hitting a wall)
        if self.grid[new_pos] == self.WALL:
            reward -= 5  # Penalty for hitting wall
            new_pos = self.krishna_pos  # Stay in current position
        
        # Update Kṛṣṇa's position
        self.grid[self.krishna_pos] = self.EMPTY
        self.krishna_pos = new_pos
        
        # Check for pellet collection
        if self.grid[new_pos] == self.PELLET:
            reward += 50
            self.score += 50
            self.pellet_positions.remove(new_pos)
        
        # Move enemies
        self._move_enemies()
        
        # Check for collisions with enemies
        if self._check_collision():
            reward -= 20
            self.lives -= 1
            if self.lives <= 0:
                done = True
        
        # Place Kṛṣṇa in new position
        self.grid[self.krishna_pos] = self.KRISHNA
        
        # Check win condition
        if self.score >= 200:
            reward += 100
            done = True
        
        return self._get_state(), reward, done, False, {}

    def _generate_walls(self):
        """
        Generate walls in the grid using a simple maze-like pattern.
        Places walls in approximately 20% of the grid cells.
        """
        # Calculate number of walls to place
        wall_count = int(0.2 * self.state_size)
        
        # Place walls randomly but ensure they don't isolate areas
        walls_placed = 0
        while walls_placed < wall_count:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            
            # Don't place walls at the edges to ensure accessibility
            if x in [0, self.grid_size-1] or y in [0, self.grid_size-1]:
                continue
                
            # Place wall if it doesn't create an isolated area
            if self._is_valid_wall_position((x, y)):
                self.grid[x, y] = self.WALL
                walls_placed += 1

    def _place_pellets(self):
        """
        Place pellets in empty cells.
        Places pellets in approximately 10% of the remaining empty cells.
        """
        empty_cells = np.where(self.grid == self.EMPTY)
        empty_positions = list(zip(empty_cells[0], empty_cells[1]))
        
        # Calculate number of pellets to place
        pellet_count = int(0.1 * len(empty_positions))
        
        # Randomly select positions for pellets
        pellet_positions = self.np_random.choice(
            len(empty_positions), 
            size=pellet_count, 
            replace=False
        )
        
        # Place pellets and store their positions
        for pos_idx in pellet_positions:
            pos = empty_positions[pos_idx]
            self.grid[pos] = self.PELLET
            self.pellet_positions.add(pos)

    def _get_random_empty_position(self):
        """
        Find a random empty cell in the grid.
        Returns:
            tuple: (x, y) coordinates of empty cell
        """
        empty_cells = np.where(self.grid == self.EMPTY)
        idx = self.np_random.integers(0, len(empty_cells[0]))
        return (empty_cells[0][idx], empty_cells[1][idx])

    def _get_new_position(self, current_pos, action):
        """
        Calculate new position based on current position and action.
        Args:
            current_pos: tuple (x, y)
            action: int (0=Up, 1=Down, 2=Left, 3=Right)
        Returns:
            tuple: New (x, y) position
        """
        x, y = current_pos
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)
        return (x, y)

    def _is_valid_wall_position(self, pos):
        """
        Check if placing a wall at the given position would create
        an isolated area.
        Args:
            pos: tuple (x, y)
        Returns:
            bool: True if wall placement is valid
        """
        # Simple check: ensure at least 2 adjacent cells are empty
        x, y = pos
        adjacent = [
            (x-1, y), (x+1, y),
            (x, y-1), (x, y+1)
        ]
        
        empty_count = 0
        for adj_x, adj_y in adjacent:
            if (0 <= adj_x < self.grid_size and 
                0 <= adj_y < self.grid_size and
                self.grid[adj_x, adj_y] == self.EMPTY):
                empty_count += 1
                
        return empty_count >= 2

    def _move_enemies(self):
        """
        Move all enemy entities according to their behavior patterns.
        - Hunter: Moves toward Kṛṣṇa
        - Greedy Bots: Move toward nearest pellet
        - Patrollers: Move in predetermined patterns
        """
        # Move Hunter (simple chase behavior)
        self.grid[self.hunter_pos] = self.EMPTY
        self.hunter_pos = self._move_towards(self.hunter_pos, self.krishna_pos)
        self.grid[self.hunter_pos] = self.HUNTER
        
        # Move Greedy Bots
        for i, pos in enumerate(self.greedy_positions):
            self.grid[pos] = self.EMPTY
            if self.pellet_positions:
                # Move toward nearest pellet
                nearest_pellet = min(self.pellet_positions, 
                                  key=lambda p: self._manhattan_distance(pos, p))
                new_pos = self._move_towards(pos, nearest_pellet)
            else:
                # If no pellets, move randomly
                new_pos = self._get_random_adjacent_position(pos)
            self.greedy_positions[i] = new_pos
            self.grid[new_pos] = self.GREEDY
        
        # Move Patrollers (simple patrol pattern)
        for i, pos in enumerate(self.patrol_positions):
            self.grid[pos] = self.EMPTY
            new_pos = self._patrol_movement(pos, i)
            self.patrol_positions[i] = new_pos
            self.grid[new_pos] = self.PATROL

    def _move_towards(self, current_pos, target_pos):
        """
        Move one step from current_pos toward target_pos.
        Args:
            current_pos: tuple (x, y)
            target_pos: tuple (x, y)
        Returns:
            tuple: New position (x, y)
        """
        x, y = current_pos
        target_x, target_y = target_pos
        
        # Determine direction with highest priority
        if abs(target_x - x) > abs(target_y - y):
            if target_x > x and self.grid[x+1, y] != self.WALL:
                return (x+1, y)
            elif target_x < x and self.grid[x-1, y] != self.WALL:
                return (x-1, y)
            elif target_y > y and self.grid[x, y+1] != self.WALL:
                return (x, y+1)
            elif target_y < y and self.grid[x, y-1] != self.WALL:
                return (x, y-1)
        else:
            if target_y > y and self.grid[x, y+1] != self.WALL:
                return (x, y+1)
            elif target_y < y and self.grid[x, y-1] != self.WALL:
                return (x, y-1)
            elif target_x > x and self.grid[x+1, y] != self.WALL:
                return (x+1, y)
            elif target_x < x and self.grid[x-1, y] != self.WALL:
                return (x-1, y)
        
        return current_pos  # If no valid move found, stay in place

    def _manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_random_adjacent_position(self, pos):
        """
        Get a random valid adjacent position.
        """
        x, y = pos
        possible_moves = [
            (x-1, y), (x+1, y),
            (x, y-1), (x, y+1)
        ]
        valid_moves = [
            move for move in possible_moves
            if (0 <= move[0] < self.grid_size and
                0 <= move[1] < self.grid_size and
                self.grid[move] != self.WALL)
        ]
        if valid_moves:
            return self.np_random.choice(valid_moves)
        return pos

    def _patrol_movement(self, pos, patrol_id):
        """
        Implement patrol movement pattern.
        Each patroller follows a different fixed pattern.
        """
        x, y = pos
        # Simple patrol patterns (clockwise or counter-clockwise)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        # Different patterns for different patrollers
        if patrol_id == 0:
            # Clockwise pattern
            direction = directions[self.steps % 4]
        else:
            # Counter-clockwise pattern
            direction = directions[(-self.steps) % 4]
            
        new_x = x + direction[0]
        new_y = y + direction[1]
        
        # Check if new position is valid
        if (0 <= new_x < self.grid_size and
            0 <= new_y < self.grid_size and
            self.grid[new_x, new_y] != self.WALL):
            return (new_x, new_y)
        return pos

    def _check_collision(self):
        """
        Check if Kṛṣṇa has collided with any enemy.
        Returns:
            bool: True if collision occurred
        """
        return (self.krishna_pos == self.hunter_pos or
                self.krishna_pos in self.greedy_positions or
                self.krishna_pos in self.patrol_positions)

    def _get_state(self):
        """
        Convert the 2D grid into a 1D state vector.
        Returns:
            numpy.array: Flattened grid state
        """
        return self.grid.flatten()

    def render(self):
        """
        Render the environment to the console (simple ASCII representation)
        """
        for row in self.grid:
            print(' '.join([str(cell) for cell in row]))
        print(f"Score: {self.score}, Lives: {self.lives}")
