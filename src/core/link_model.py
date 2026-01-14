import numpy as np
from typing import List


def assign_beacon_links(num_beacons: int, los_probability: float = 0.5) -> List[int]:
    """
    Assign LoS/NLoS status to beacons.
    
    Args:
        num_beacons: Number of beacons to assign links for
        los_probability: Probability of a beacon being LoS (default: 0.5)
    
    Returns:
        List of link statuses where 1 = LoS, 0 = NLoS
    """
    links = np.random.binomial(1, los_probability, num_beacons)
    return links.tolist()
