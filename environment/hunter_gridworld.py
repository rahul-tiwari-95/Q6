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
from typing import Tuple, List, Set, Optional, Dict, Any

from .entities import Krishna, Hunter, GreedyBot, Patroller
from .entities.hunter_progressive import create_hunter_for_phase
from utils.success_scoring import EpisodeSuccessScorer, SuccessTier

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

    def __init__(self, difficulty_level: int = 1, win_condition: str = 'pellets', curriculum_phase: int = 1):
        """
        Initialize the HunterGridworld environment.
        
        Args:
            difficulty_level (int): Controls enemy count for curriculum learning
                1 = Easy (2 Greedy bots only)
                2 = Medium (2 Greedy + 1 Patroller)
                3 = Hard (2 Greedy + 2 Patrollers)
                4 = Expert (2 Greedy + 2 Patrollers + 1 Progressive Hunter)
            win_condition (str): Determines what constitutes a "win"
                'pellets' = Win by collecting target number of pellets (DEFAULT, RECOMMENDED)
                'score' = Win by reaching target score threshold
                'survival' = Win by surviving for target number of steps
                'pellets_and_survival' = Win by both pellets AND survival
            curriculum_phase (int): Progressive Hunter difficulty (1-4)
                1 = Random Hunter (Phase 1)
                2 = Greedy Hunter (Phase 2)
                3 = Smart Greedy Hunter (Phase 3)
                4 = A* Hunter (Phase 4)
        """
        super(HunterGridworld, self).__init__()

        # Grid dimensions
        self.grid_size = 25
        self.state_size = self.grid_size * self.grid_size
        
        # CURRICULUM LEARNING: Difficulty determines enemy count
        self.difficulty_level = difficulty_level
        
        # V0.5 NEW: Progressive Hunter difficulty
        self.curriculum_phase = curriculum_phase
        
        # GENERALIZED WIN CONDITION: Flexible across experiments
        self.win_condition_type = win_condition
        self.target_pellets = 4  # Pellets needed to win
        self.target_score = 200  # Score needed to win (if using score-based)
        self.target_survival_steps = 500  # Steps needed to survive (if using survival-based)
        
        # V0.5 NEW: Success Scoring System
        self.success_scorer = EpisodeSuccessScorer()

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

        # Initialize entities
        self.krishna = None
        self.hunter = None
        self.greedy_bots = []
        self.patrollers = []
        
        # Game state variables
        self.grid = None
        self.pellet_positions = set()
        self.steps = 0
        self.walls_hit = 0  # V0.5 NEW: Track wall collisions for strategy score

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        Returns:
            state: Initial state of the environment
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset counters
        self.steps = 0
        self.walls_hit = 0  # V0.5 NEW
        
        # Initialize empty grid
        self.grid = np.full((self.grid_size, self.grid_size), self.EMPTY)
        
        # Generate walls (approximately 20% of the grid)
        self._generate_walls()
        
        # Place pellets (approximately 10% of remaining space)
        self._place_pellets()
        
        # Create and place Kṛṣṇa
        krishna_pos = self._get_random_empty_position()
        self.krishna = Krishna(krishna_pos, self.grid_size)
        self.grid[krishna_pos] = self.KRISHNA
        
        # ===== CURRICULUM LEARNING: Create enemies based on difficulty =====
        
        # Level 1 (Easy): Only 2 Greedy Bots
        # Level 2 (Medium): 2 Greedy Bots + 1 Patroller
        # Level 3 (Hard): 2 Greedy Bots + 2 Patrollers
        # Level 4 (Expert): 2 Greedy Bots + 2 Patrollers + 1 Hunter (FULL GAME)
        
        # Always create 2 Greedy Bots (all difficulty levels)
        self.greedy_bots = []
        for _ in range(2):
            pos = self._get_random_empty_position()
            greedy_bot = GreedyBot(pos, self.grid_size)
            self.greedy_bots.append(greedy_bot)
            self.grid[pos] = self.GREEDY
        
        # Create Patrollers based on difficulty
        self.patrollers = []
        if self.difficulty_level >= 2:
            # Level 2+: Add 1 Patroller
            pos = self._get_random_empty_position()
            patroller = Patroller(pos, self.grid_size, patrol_id=0)
            self.patrollers.append(patroller)
            self.grid[pos] = self.PATROL
            
        if self.difficulty_level >= 3:
            # Level 3+: Add 2nd Patroller
            pos = self._get_random_empty_position()
            patroller = Patroller(pos, self.grid_size, patrol_id=1)
            self.patrollers.append(patroller)
            self.grid[pos] = self.PATROL
        
        # Create Hunter only at highest difficulty
        # V0.5 NEW: Uses progressive Hunter (difficulty varies by curriculum_phase)
        self.hunter = None
        if self.difficulty_level >= 4:
            # Level 4 (Expert): Add Progressive Hunter
            hunter_pos = self._get_random_empty_position()
            self.hunter = create_hunter_for_phase(self.curriculum_phase, hunter_pos, self.grid_size)
            self.grid[hunter_pos] = self.HUNTER
        
        self.steps = 0
        
        return self._get_state(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        This is the core game loop where one action is taken and the world responds.
        
        The reward structure is carefully designed to:
        - Encourage pellet collection (large positive reward)
        - Discourage getting caught (moderate negative reward)
        - Penalize wall hits (small negative reward)
        - Give minimal survival penalty (very small negative per step)
        
        Args:
            action: int (0=Up, 1=Down, 2=Left, 3=Right)
        Returns:
            state: Current state of environment
            reward: Reward from the action
            done: Whether the episode has ended
            truncated: Whether episode was artificially terminated
            info: Additional information
        """
        self.steps += 1
        # IMPROVED REWARD STRUCTURE (v0.4):
        # -0.001 per step (was -0.01) - minimal survival penalty
        # This encourages efficiency without making survival itself punishing
        # With 1000 steps max: -0.001/step = -1 penalty (very reasonable)
        reward = -0.001
        done = False
        
        # Clear Krishna's current position from the grid
        # (so we can update it to the new position)
        self.grid[self.krishna.position] = self.EMPTY
        
        # Move Krishna based on the action
        # Note: If the move is blocked by a wall, Krishna stays in place
        old_pos = self.krishna.position
        new_pos = self.krishna.move(action, self.grid)
        
        # If Krishna tried to move but hit a wall (position unchanged)
        # Give a small penalty to discourage wall-hitting behavior
        if new_pos == old_pos and self.grid[new_pos] != self.WALL:
            reward -= 5  # Wall hit penalty
            self.walls_hit += 1  # V0.5 NEW: Track for strategy scoring
            
        # Check for pellet collection
        # Pellets are the main reward mechanism - agent must collect 4 to win (4 * 50 = 200 points)
        if self.grid[new_pos] == self.PELLET:
            self.krishna.collect_pellet()
            reward += 50  # Major reward for collecting pellet - this is what we want to encourage
            self.pellet_positions.remove(new_pos)
        
        # Update Krishna's position on grid
        self.grid[new_pos] = self.KRISHNA
        
        # ===== MOVE ALL ENEMIES =====
        # Each enemy type has its own behavior:
        # - Hunter: Actively chases Krishna using pathfinding (only at difficulty 4)
        # - Greedy Bots: Chase nearest pellet (all difficulties)
        # - Patrollers: Follow fixed patrol routes (difficulty 2+)
        
        # Move Hunter (only exists at difficulty level 4)
        if self.hunter is not None:
            self.grid[self.hunter.position] = self.EMPTY
            self.hunter.update(self.grid, self.krishna.position, self.krishna.previous_positions)
            self.grid[self.hunter.position] = self.HUNTER
        
        # Move Greedy Bots (always present)
        for bot in self.greedy_bots:
            self.grid[bot.position] = self.EMPTY
            bot.update(self.grid, self.pellet_positions)
            self.grid[bot.position] = self.GREEDY
            
        # Move Patrollers (only at difficulty 2+)
        for patroller in self.patrollers:
            self.grid[patroller.position] = self.EMPTY
            patroller.update(self.grid)
            self.grid[patroller.position] = self.PATROL
        
        # Check for collisions with enemies
        # If Krishna collides, it takes damage (loses a life)
        if self._check_collision():
            reward -= 10  # V0.5 CHANGED: Reduced from -20 to -10 (softer learning signal)
            if not self.krishna.take_damage():  # Returns False if no lives left
                done = True  # Episode ends if all lives exhausted
        
        # Update Krishna's internal state (like invulnerability timer)
        self.krishna.update()
        
        # ===== DECOUPLED WIN CONDITION AND EPISODE TERMINATION =====
        # NEW PHILOSOPHY (v0.4):
        # 1. Score tracks performance metrics (pellets, penalties, etc.)
        # 2. Win condition determines TASK COMPLETION (did agent achieve goal?)
        # 3. Episode terminates when:
        #    - Task completed successfully (win condition met) → Give success bonus
        #    - Agent dies (no lives left) → Episode failure
        #    - Max steps reached → Truncated (timeout)
        #
        # This prevents "reward hacking" where agent optimizes score instead of task.
        
        # Check win condition (task completion)
        if self.check_win_condition():
            reward += 100  # Success bonus for completing the objective
            done = True
        
        # Check death condition (all lives exhausted)
        # Note: This is already handled above in collision check,
        # but keeping logic clear here
        
        # Check if maximum steps reached (timeout)
        truncated = self.steps >= 1000
        
        return self._get_state(), reward, done, truncated, self._get_info()

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
        
        # Safety check: if no empty cells, raise error with diagnostic info
        if len(empty_cells[0]) == 0:
            raise RuntimeError(
                f"No empty positions available in grid!\n"
                f"Grid contents:\n"
                f"  Walls: {np.sum(self.grid == self.WALL)}\n"
                f"  Pellets: {np.sum(self.grid == self.PELLET)}\n"
                f"  Krishna: {np.sum(self.grid == self.KRISHNA)}\n"
                f"  Hunter: {np.sum(self.grid == self.HUNTER)}\n"
                f"  Greedy: {np.sum(self.grid == self.GREEDY)}\n"
                f"  Patrol: {np.sum(self.grid == self.PATROL)}\n"
                f"  Empty: {np.sum(self.grid == self.EMPTY)}"
            )
        
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

    def _check_collision(self) -> bool:
        """
        Check if Kṛṣṇa has collided with any enemy.
        Returns:
            bool: True if collision occurred
        """
        # Don't check collisions if Krishna is invulnerable
        if self.krishna.invulnerable:
            return False
        
        # Check Hunter collision (only if Hunter exists at difficulty level 4)
        hunter_collision = (self.hunter is not None and 
                           self.krishna.position == self.hunter.position)
            
        return (hunter_collision or
                any(self.krishna.position == bot.position for bot in self.greedy_bots) or
                any(self.krishna.position == patroller.position for patroller in self.patrollers))

    def _get_state(self) -> np.ndarray:
        """
        Convert the 2D grid into a 1D state vector.
        Returns:
            numpy.array: Flattened grid state
        """
        return self.grid.flatten()

    def _get_info(self) -> Dict[str, Any]:
        """
        Get current game state information.
        Returns:
            dict: Dictionary containing current game state info
        """
        # Base info
        info = {
            'score': self.krishna.score,
            'lives': self.krishna.lives,
            'steps': self.steps,
            'pellets_remaining': len(self.pellet_positions),
            'pellets_collected': self.krishna.pellets_collected,
            'krishna_position': self.krishna.position,
            'hunter_position': self.hunter.position if self.hunter else None,
            'is_invulnerable': self.krishna.invulnerable,
            'times_caught': 3 - self.krishna.lives,  # V0.5 NEW: For success scoring
            'steps_taken': self.steps,  # V0.5 NEW: For success scoring
            'walls_hit': self.walls_hit,  # V0.5 NEW: For success scoring
            'phase': self.curriculum_phase,  # V0.5 NEW: For success scoring
        }
        
        # V0.5 NEW: Calculate success score if episode has data
        if self.steps > 0:
            total_score, component_scores = self.success_scorer.calculate_score(info)
            tier = self.success_scorer.get_tier(total_score)
            
            info.update({
                'success_score': total_score,
                'success_components': component_scores,
                'success_tier': tier.grade,
                'success_tier_label': tier.label,
                'success_tier_emoji': tier.emoji
            })
        else:
            # Episode just started, no score yet
            info.update({
                'success_score': 0.0,
                'success_components': {},
                'success_tier': 'F',
                'success_tier_label': 'FAILURE',
                'success_tier_emoji': '❌'
            })
        
        return info
    
    def check_win_condition(self) -> bool:
        """
        Check if the agent has met the win condition.
        This is a GENERALIZED function that works across different experiments.
        
        PHILOSOPHY:
        -----------
        Win conditions should be based on TASK COMPLETION, not scoring mechanics.
        Score is accumulated through gameplay but doesn't always reflect success.
        
        Win Condition Types:
        - 'pellets': Win by collecting target pellets (RECOMMENDED - task-focused)
        - 'score': Win by reaching score threshold (traditional but can be gamed)
        - 'survival': Win by surviving for target steps (endurance-focused)
        - 'pellets_and_survival': Win by both collecting pellets AND surviving
        
        Returns:
            bool: True if win condition met, False otherwise
        """
        if self.win_condition_type == 'pellets':
            # RECOMMENDED: Task-based win condition
            # Win by collecting the target number of pellets
            return self.krishna.pellets_collected >= self.target_pellets
            
        elif self.win_condition_type == 'score':
            # Traditional score-based (can lead to reward hacking)
            return self.krishna.score >= self.target_score
            
        elif self.win_condition_type == 'survival':
            # Survival-based: Win by lasting long enough
            return self.steps >= self.target_survival_steps
            
        elif self.win_condition_type == 'pellets_and_survival':
            # Combined: Must collect pellets AND survive
            return (self.krishna.pellets_collected >= self.target_pellets and
                   self.steps >= self.target_survival_steps)
        
        # Default fallback
        return False

    def render(self) -> None:
        """
        Render the environment to the console (simple ASCII representation)
        """
        # Create symbol mapping for better visualization
        symbols = {
            self.WALL: '█',
            self.PELLET: '·',
            self.KRISHNA: 'K',
            self.HUNTER: 'H',
            self.GREEDY: 'G',
            self.PATROL: 'P',
            self.EMPTY: ' '
        }
        
        # Print the grid
        print('╔' + '═' * (self.grid_size * 2 - 1) + '╗')
        for row in self.grid:
            print('║' + ' '.join(symbols[cell] for cell in row) + '║')
        print('╚' + '═' * (self.grid_size * 2 - 1) + '╝')
        
        # Print game stats
        print(f"Score: {self.krishna.score} | Lives: {self.krishna.lives} | Steps: {self.steps}")
        if self.krishna.invulnerable:
            print(f"Invulnerable for {self.krishna.invulnerable_timer} steps")
