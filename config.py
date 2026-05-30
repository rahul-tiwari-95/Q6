"""
Q6 Project Configuration
=========================
Centralized hyperparameters, grid constants, and training settings.
Import this module instead of hard-coding values across files.
"""

from typing import Tuple

# === GRID & ENVIRONMENT ===
GRID_SIZE: int = 25
STATE_SIZE: int = GRID_SIZE * GRID_SIZE  # 625
ACTION_SIZE: int = 4  # Up, Down, Left, Right

# Grid cell constants
WALL = 0
PELLET = 1
KRISHNA = 2
HUNTER = 3
GREEDY = 4
PATROL = 5
EMPTY = 6

# Grid generation
WALL_DENSITY: float = 0.20   # ~20% of grid is walls
PELLET_DENSITY: float = 0.10  # ~10% of remaining space

# === REWARDS ===
REWARD_PELLET: float = 50.0
REWARD_WIN: float = 100.0
REWARD_CAUGHT: float = -10.0    # Penalty for being caught by enemy (v0.5: reduced from -20)
REWARD_WALL_HIT: float = -5.0   # Penalty for hitting a wall
REWARD_STEP: float = -0.001    # Per-step survival penalty (v0.3: was -0.01, v0.2: was -0.1)

# === WIN/LOSS CONDITIONS ===
TARGET_PELLETS: int = 4         # Pellets to collect to win
INITIAL_LIVES: int = 3
INVULNERABILITY_FRAMES: int = 30  # Frames of invulnerability after taking damage
MAX_STEPS_PER_EPISODE: int = 1000


# === DQN AGENT ===
HIDDEN_SIZES: Tuple[int, ...] = (256, 128)
LEARNING_RATE: float = 1e-4
GAMMA: float = 0.99            # Discount factor
TAU: float = 1e-3              # Soft update parameter for target network
BATCH_SIZE: int = 64
BUFFER_SIZE: int = 100_000
UPDATE_EVERY: int = 4          # Learn every N steps

# Epsilon-greedy exploration
EPSILON_START: float = 1.0
EPSILON_MIN: float = 0.05
EPSILON_DECAY: float = 0.9999  # Per-episode decay rate
# Phase-specific epsilon resets
PHASE_EPSILON_RESET: dict = {
    1: 1.0,   # Random Hunter: full exploration
    2: 0.5,   # Greedy Hunter: balanced
    3: 0.3,   # Smart Greedy: mostly exploit
    4: 0.2,   # A* Hunter: strategic exploration
}


# === CURRICULUM LEARNING ===
# Phase progression: hunter difficulty increases gradually
CURRICULUM_PHASES: dict = {
    1: {"enemies": "2 Greedy Bots only", "hunter": "Random Hunter"},
    2: {"enemies": "2 Greedy + 1 Patroller", "hunter": "Greedy Hunter"},
    3: {"enemies": "2 Greedy + 2 Patrollers", "hunter": "Smart Greedy Hunter"},
    4: {"enemies": "2 Greedy + 2 Patrollers + 1 Hunter", "hunter": "A* Hunter"},
}

# === TRAINING ===
DEFAULT_EPISODES: int = 18_000  # Must match sum(PHASE_EPISODES): 3000+3000+3000+9000
DEFAULT_VERBOSE_EVERY: int = 10
DEFAULT_CHECKPOINT_EVERY: int = 100

# Phase episode allocation [phase1, phase2, phase3, phase4]
PHASE_EPISODES: Tuple[int, ...] = (3000, 3000, 3000, 9000)


# === SUCCESS SCORING (v0.5) ===
WEIGHT_OBJECTIVES: float = 0.40
WEIGHT_SURVIVAL: float = 0.30
WEIGHT_EFFICIENCY: float = 0.20
WEIGHT_STRATEGY: float = 0.10


# === LOGGING ===
LOG_DIR: str = "training_runs"
CHECKPOINT_DIR: str = "checkpoints"
PLOT_DIR: str = "plots"
MOVING_AVG_WINDOW: int = 100  # Window for moving average of scores
