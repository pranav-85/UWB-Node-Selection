import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from localization.trilateration import uwb_trilateration_epoch
from config import EPSILON, NUM_BEACONS, ER_TH, MD_TH


def compute_reward(agent_pos, beacon_positions, los_flags, battery_levels):
    """
    Compute reward based on localization error and beacon battery deviation.

    Args:
        agent_pos: (x, y) ground-truth position of the agent
        beacon_positions: List of (x, y) positions of selected beacons
        los_flags: List of booleans indicating LOS/NLOS status for each beacon
        battery_levels: List of current battery levels for all beacons

    Returns:
        reward: Scalar reward value
    """

    result = uwb_trilateration_epoch(
        target_pos=agent_pos,
        beacon_positions=beacon_positions,
        los_flags=los_flags
    )

    ER_t = result["localization_error"]  # localization error

    
    battery_levels = np.array(battery_levels, dtype=float)

    B_mean = np.mean(battery_levels)

    # Normalized mean-squared deviation (as defined)
    MD_t = (1.0 / (NUM_BEACONS - 1)) * np.sum(
        ((battery_levels - B_mean) / (B_mean + EPSILON)) ** 2
    )

    
    reward_magnitude = (ER_t + EPSILON) * (MD_t + EPSILON)

    if ER_t <= ER_TH and MD_t <= MD_TH:
        reward = 1.0 / reward_magnitude
    else:
        reward = -reward_magnitude

    return reward
