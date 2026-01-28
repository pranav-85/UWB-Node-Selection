import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from localization.trilateration import uwb_trilateration_epoch
from config import EPSILON, NUM_BEACONS, ER_TH, MD_TH


# Reward function weights (trade-off parameters)
ALPHA = 1.0   # Weight for localization error term
BETA = 0.3    # Weight for battery deviation term


def compute_reward(agent_pos, beacon_positions, los_flags, battery_levels):
    """
    Compute weighted, bounded, additive reward.
    
    Reward function: R_t = -α · (ER_t / ER_th) - β · (MD_t / MD_th)
    
    This is a bounded, additive reward that penalizes:
    1. Localization error (ER_t) with weight α
    2. Battery mean deviation (MD_t) with weight β
    
    Args:
        agent_pos: (x, y) ground-truth position of the agent
        beacon_positions: List of (x, y) positions of selected beacons
        los_flags: List of booleans indicating LOS/NLOS status for each beacon
        battery_levels: List of current battery levels for all beacons

    Returns:
        reward: Scalar reward value (typically in range [-2, 0] with default weights)
    """

    # Compute localization error
    result = uwb_trilateration_epoch(
        target_pos=agent_pos,
        beacon_positions=beacon_positions,
        los_flags=los_flags
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
