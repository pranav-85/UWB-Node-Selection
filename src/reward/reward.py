"""
Reward function for UWB node selection RL training.

Supports both simple noise and CIR-based distance measurement models.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from localization.trilateration import uwb_trilateration_epoch
from config import EPSILON, NUM_BEACONS, ER_TH, MD_TH


# Reward function weights (trade-off parameters)
ALPHA = 1.0   # Weight for localization error term
BETA = 0.3    # Weight for battery deviation term

# Global flag to use CIR model (can be set via set_cir_mode())
_USE_CIR_MODEL = False
_CIR_CONFIG = None


def set_cir_mode(use_cir: bool, cir_config = None):
    """
    Enable or disable CIR-based distance measurement for reward computation.
    
    Args:
        use_cir: If True, use CIR model; if False, use simple noise
        cir_config: Optional CIRConfig object (uses defaults if None)
    """
    global _USE_CIR_MODEL, _CIR_CONFIG
    _USE_CIR_MODEL = use_cir
    _CIR_CONFIG = cir_config
    if use_cir:
        print(f"[OK] CIR-based distance model enabled")
    else:
        print(f"[OK] Simple noise distance model enabled")


def get_cir_mode() -> bool:
    """Get current CIR mode setting."""
    return _USE_CIR_MODEL


def compute_reward(agent_pos, beacon_positions, los_flags, battery_levels):
    """
    Compute weighted, bounded, additive reward.
    
    Reward function: R_t = -α · (ER_t / ER_th) - β · (MD_t / MD_th)
    
    This is a bounded, additive reward that penalizes:
    1. Localization error (ER_t) with weight α
    2. Battery mean deviation (MD_t) with weight β
    
    Supports both simple noise and CIR-based distance measurement models.
    Use set_cir_mode() to switch between models.
    
    Args:
        agent_pos: (x, y) ground-truth position of the agent
        beacon_positions: List of (x, y) positions of selected beacons
        los_flags: List of booleans indicating LOS/NLOS status for each beacon
        battery_levels: List of current battery levels for all beacons

    Returns:
        reward: Scalar reward value (typically in range [-2, 0] with default weights)
    """

    # Compute localization error using selected distance model
    result = uwb_trilateration_epoch(
        target_pos=agent_pos,
        beacon_positions=beacon_positions,
        los_flags=los_flags,
        use_cir=_USE_CIR_MODEL,
        cir_config=_CIR_CONFIG
    )

    ER_t = result["localization_error"]  # localization error

    # Compute battery mean deviation
    battery_levels = np.array(battery_levels, dtype=float)
    B_mean = np.mean(battery_levels)

    # Normalized mean-squared deviation
    MD_t = (1.0 / (NUM_BEACONS - 1)) * np.sum(
        ((battery_levels - B_mean) / (B_mean + EPSILON)) ** 2
    )

    # Weighted, bounded, additive reward
    # R_t = -α · (ER_t / ER_th) - β · (MD_t / MD_th)
    reward = -ALPHA * (ER_t / (ER_TH + EPSILON)) - BETA * (MD_t / (MD_TH + EPSILON))

    return reward
