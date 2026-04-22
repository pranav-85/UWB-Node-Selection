"""
Reward function for UWB node selection RL training.

Supports both simple noise and CIR-based distance measurement models.
"""

import sys
from pathlib import Path
import numpy as np
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EPSILON, NUM_BEACONS, ER_TH, MD_TH
from localization.gdop import compute_weighted_gdop


# Reward function weights (trade-off parameters)
ALPHA = 0.6   # Geometry quality penalty
BETA = 0.3    # Battery deviation penalty
GAMMA = 0.1   # LoS/NLoS penalty
DELTA = 0.1   # Optional GDOP penalty

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


def _triangle_area(a, b, c) -> float:
    """Compute 2D triangle area from 3 points."""
    return abs(
        0.5 * (
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
    )


def _geometry_penalty(beacon_positions: np.ndarray) -> float:
    """Compute normalized geometry penalty from selected beacon spread."""
    if len(beacon_positions) < 3:
        return 1.0

    areas = []
    for i, j, k in combinations(range(len(beacon_positions)), 3):
        areas.append(_triangle_area(beacon_positions[i], beacon_positions[j], beacon_positions[k]))

    max_area = max(areas) if areas else 0.0
    pairwise = []
    for i, j in combinations(range(len(beacon_positions)), 2):
        pairwise.append(float(np.linalg.norm(beacon_positions[i] - beacon_positions[j])))
    max_pairwise = max(pairwise) if pairwise else 1.0
    norm_area = max_area / (max_pairwise ** 2 + EPSILON)
    spread_score = np.clip(norm_area * 4.0, 0.0, 1.0)
    return 1.0 - spread_score


def _gdop_penalty(beacon_positions: np.ndarray, los_flags: list) -> float:
    """Optional GDOP-based penalty from observable geometry and LoS state."""
    if len(beacon_positions) < 3:
        return 1.0

    agent_estimate = np.mean(beacon_positions, axis=0)
    gdop = compute_weighted_gdop(agent_estimate, beacon_positions, los_flags)
    if not np.isfinite(gdop):
        return 1.0
    return float(np.clip(gdop / 10.0, 0.0, 1.0))


def compute_reward(agent_pos, beacon_positions, los_flags, battery_levels):
    """
    Compute weighted, bounded, additive reward from observable features only.

    Ground-truth agent position is accepted for backward compatibility but is
    intentionally not used in reward computation.
    
    Args:
        agent_pos: Unused (kept for backward-compatible call signature)
        beacon_positions: List of (x, y) positions of selected beacons
        los_flags: List of booleans indicating LOS/NLOS status for each beacon
        battery_levels: List of current battery levels for all beacons

    Returns:
        reward: Scalar reward value based on battery, geometry, and LoS quality.
    """

    beacon_positions = np.array(beacon_positions, dtype=float)

    # Compute battery mean deviation
    battery_levels = np.array(battery_levels, dtype=float)
    battery_levels = battery_levels / 100.0
    B_mean = np.mean(battery_levels)

    # Normalized mean-squared deviation
    MD_t = (1.0 / (NUM_BEACONS - 1)) * np.sum(
        ((battery_levels - B_mean) / (B_mean + EPSILON)) ** 2
    )

    geometry_pen = _geometry_penalty(beacon_positions)
    los_pen = 1.0 - float(np.mean(np.array(los_flags, dtype=float)))
    gdop_pen = _gdop_penalty(beacon_positions, los_flags)

    # Normalized and bounded additive reward from observable terms.
    reward = (
        -ALPHA * geometry_pen
        -BETA * (MD_t / (MD_TH + EPSILON))
        -GAMMA * los_pen
        -DELTA * gdop_pen
    )

    return reward
