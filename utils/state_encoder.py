"""
State Encoder — convert flat grid state to multi-channel tensor for CNN.

The HunterGridworld returns a flat (625,) int32 vector with values 0..6:
    0=WALL, 1=PELLET, 2=KRISHNA, 3=HUNTER, 4=GREEDY, 5=PATROL, 6=EMPTY

A CNN needs spatial structure. We encode the grid as a (6, 25, 25) binary
tensor — one channel per non-empty entity type. EMPTY is implicit (all
channels zero at that cell).

Channels (fixed order):
    0: walls
    1: pellets
    2: krishna
    3: hunter
    4: greedy bots
    5: patrollers
"""

from __future__ import annotations

import numpy as np

from config import GRID_SIZE, WALL, PELLET, KRISHNA, HUNTER, GREEDY, PATROL

NUM_CHANNELS: int = 6

# Maps cell value -> channel index. EMPTY (6) is intentionally absent.
_CELL_TO_CHANNEL: dict[int, int] = {
    WALL:    0,
    PELLET:  1,
    KRISHNA: 2,
    HUNTER:  3,
    GREEDY:  4,
    PATROL:  5,
}


def encode_state(flat_state: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Convert a flat (grid_size*grid_size,) state vector into a
    (NUM_CHANNELS, grid_size, grid_size) float32 binary tensor.

    Args:
        flat_state: 1-D array of cell values (ints 0..6).
        grid_size:  side length of the grid (default GRID_SIZE).

    Returns:
        np.ndarray of shape (6, grid_size, grid_size), dtype float32, values in {0., 1.}.
    """
    if flat_state.ndim != 1:
        raise ValueError(f"flat_state must be 1-D, got shape {flat_state.shape}")
    expected = grid_size * grid_size
    if flat_state.shape[0] != expected:
        raise ValueError(f"flat_state length {flat_state.shape[0]} != {expected}")

    grid = flat_state.reshape(grid_size, grid_size)
    out = np.zeros((NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
    for cell_value, channel in _CELL_TO_CHANNEL.items():
        out[channel] = (grid == cell_value).astype(np.float32)
    return out


def encode_batch(flat_states: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Vectorized batch encoder. Accepts (B, grid_size*grid_size) and returns
    (B, NUM_CHANNELS, grid_size, grid_size) float32.
    """
    if flat_states.ndim != 2:
        raise ValueError(f"flat_states must be 2-D, got shape {flat_states.shape}")
    B = flat_states.shape[0]
    grids = flat_states.reshape(B, grid_size, grid_size)
    out = np.zeros((B, NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
    for cell_value, channel in _CELL_TO_CHANNEL.items():
        out[:, channel] = (grids == cell_value).astype(np.float32)
    return out
