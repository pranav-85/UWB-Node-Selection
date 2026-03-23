import numpy as np


def compute_weighted_gdop(agent_estimate,
                          beacon_positions,
                          los_flags,
                          sigma_los=0.05,
                          sigma_nlos=0.2):
    """
    Compute real-time Weighted GDOP (WGDOP).

    Uses estimated position only (no ground truth).

    Args:
        agent_estimate: (x_hat, y_hat)
        beacon_positions: list of (x, y)
        los_flags: list of booleans
        sigma_los: std deviation for LoS
        sigma_nlos: std deviation for NLoS

    Returns:
        wgdop (float)
    """

    x_hat, y_hat = agent_estimate
    H = []
    W_diag = []

    for (x_b, y_b), los in zip(beacon_positions, los_flags):

        dx = x_hat - x_b
        dy = y_hat - y_b
        d = np.sqrt(dx**2 + dy**2)

        if d < 1e-6:
            return np.inf

        H.append([dx / d, dy / d])

        sigma = sigma_los if los else sigma_nlos
        W_diag.append(1.0 / (sigma ** 2))

    H = np.array(H)
    W = np.diag(W_diag)

    try:
        Q = np.linalg.inv(H.T @ W @ H)
        wgdop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        wgdop = np.inf

    return wgdop
