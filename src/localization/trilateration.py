"""
Trilateration-based localization simulation with noisy UWB distance measurements.

Implements:
- True distance computation
- LoS / NLoS noise model (simple and CIR-based)
- Linearized least-squares trilateration (2D)
- Localization error computation

Assumptions:
- Beacon positions are fixed and known
- Target ground truth is known to simulator
- Exactly three beacons are used per localization
- Noise is added only to distance measurements

CIR Integration:
- Supports both simple noise model and CIR-based model
- Use compute_distances() with use_cir=True for CIR model
- Use compute_distances() with use_cir=False for simple noise
"""

import numpy as np



def noisy_distance(d_true: float, los: bool) -> float:
    """
    Generate noisy distance measurement.

    LoS  -> small zero-mean Gaussian noise
    NLoS -> positive bias + larger Gaussian noise
    """
    if los:
        noise = np.random.normal(0.0, 0.05)      # meters
    else:
        bias = np.random.uniform(0.3, 1.0)       # meters
        noise = np.random.normal(0.0, 0.2) + bias

    return d_true + noise



def compute_noisy_distances(
    target_pos,
    beacon_positions,
    los_flags
):
    """
    Compute noisy distances from target to selected beacons.

    target_pos       : (x_t, y_t)
    beacon_positions : [(x_i, y_i), (x_j, y_j), (x_k, y_k)]
    los_flags        : [True/False, True/False, True/False]
    """
    x_t, y_t = target_pos
    distances = []

    for (x_b, y_b), los in zip(beacon_positions, los_flags):
        d_true = np.sqrt((x_t - x_b)**2 + (y_t - y_b)**2)
        d_noisy = noisy_distance(d_true, los)
        distances.append(d_noisy)

    return distances


def compute_distances(
    target_pos,
    beacon_positions,
    los_flags,
    use_cir: bool = False,
    cir_config = None
):
    """
    Compute distances from target to selected beacons.
    
    Supports both simple noise model and CIR-based model.
    
    Args:
        target_pos: (x_t, y_t) target position
        beacon_positions: [(x_i, y_i), ...] beacon positions  
        los_flags: [True/False, ...] LoS/NLoS flags
        use_cir: If True, use CIR model; if False, use simple noise
        cir_config: CIRConfig object for CIR model (uses defaults if None)
    
    Returns:
        List of estimated distances
    """
    if use_cir:
        # Use CIR-based distances
        try:
            from localization.cir_model import compute_cir_distances, FAST_CIR_CONFIG
            if cir_config is None:
                cir_config = FAST_CIR_CONFIG
            distances = compute_cir_distances(
                target_pos, beacon_positions, los_flags, config=cir_config
            )
        except ImportError as e:
            # Fallback to simple noise if CIR module not available
            print(f"Warning: CIR module not imported ({e}). Falling back to simple noise model.")
            distances = compute_noisy_distances(target_pos, beacon_positions, los_flags)
    else:
        # Use simple noise model
        distances = compute_noisy_distances(target_pos, beacon_positions, los_flags)
    
    return distances



def trilateration_2d(beacon_positions, distances):
    """
    Solve 2D trilateration using linearized least squares.

    beacon_positions : [(x_i, y_i), (x_j, y_j), (x_k, y_k)]
    distances        : [d_i, d_j, d_k]
    """
    (x_i, y_i), (x_j, y_j), (x_k, y_k) = beacon_positions
    d_i, d_j, d_k = distances

    A = np.array([
        [2 * (x_j - x_i), 2 * (y_j - y_i)],
        [2 * (x_k - x_i), 2 * (y_k - y_i)]
    ])

    b = np.array([
        x_j**2 + y_j**2 - x_i**2 - y_i**2 + d_i**2 - d_j**2,
        x_k**2 + y_k**2 - x_i**2 - y_i**2 + d_i**2 - d_k**2
    ])

    # Least-squares solution (robust to ill-conditioning)
    est_pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return est_pos[0], est_pos[1]


def localization_error(true_pos, est_pos):
    """
    Squared localization error (ER_t).
    """
    x_t, y_t = true_pos
    x_hat, y_hat = est_pos

    return (x_t - x_hat)**2 + (y_t - y_hat)**2



def uwb_trilateration_epoch(
    target_pos,
    beacon_positions,
    los_flags,
    use_cir: bool = False,
    cir_config = None
):
    """
    Complete localization pipeline for one time step.
    
    Args:
        target_pos: Ground truth target position (x, y)
        beacon_positions: List of beacon positions
        los_flags: List of LoS/NLoS flags
        use_cir: If True, use CIR model; if False, use simple noise
        cir_config: CIRConfig object for CIR model
    
    Returns:
        Dictionary with localization results
    """
    distances = compute_distances(
        target_pos,
        beacon_positions,
        los_flags,
        use_cir=use_cir,
        cir_config=cir_config
    )

    est_pos = trilateration_2d(
        beacon_positions,
        distances
    )

    error = localization_error(
        target_pos,
        est_pos
    )

    return {
        "true_position": target_pos,
        "estimated_position": est_pos,
        "distances": distances,
        "localization_error": np.sqrt(error)
    }



if __name__ == "__main__":
    np.random.seed(42)

    # Ground truth target position
    target_position = (4.0, 3.0)

    # Selected beacon positions (i, j, k)
    beacons = [
        (0.0, 0.0),
        (8.0, 0.0),
        (4.0, 7.0)
    ]

    # LoS / NLoS condition per beacon
    los_conditions = [True, False, True]

    result = uwb_trilateration_epoch(
        target_position,
        beacons,
        los_conditions
    )

    print("True Position      :", result["true_position"])
    print("Estimated Position :", result["estimated_position"])
    print("Noisy Distances    :", result["distances"])
    print("Localization Error :", result["localization_error"])
