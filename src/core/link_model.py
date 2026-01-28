import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path


def generate_los_map(grid_width: int, grid_height: int, num_beacons: int, 
                     los_probability: float = 0.5, grid_resolution: float = 1.0) -> Dict[Tuple[float, float], List[int]]:
    """
    Pre-compute LoS/NLoS links for all possible agent positions on a discrete grid.
    
    Args:
        grid_width: Width of the environment grid
        grid_height: Height of the environment grid
        num_beacons: Number of beacons
        los_probability: Probability of LoS for each beacon
        grid_resolution: Resolution of the discretized grid (e.g., 1.0 for integer positions)
    
    Returns:
        Dictionary mapping discretized (x, y) positions to list of link statuses
    """
    los_map = {}
    
    # Use numpy arange for floating point grid resolution
    x_positions = np.arange(0, grid_width + grid_resolution, grid_resolution)
    y_positions = np.arange(0, grid_height + grid_resolution, grid_resolution)
    
    for x in x_positions:
        for y in y_positions:
            # Assign links for this position (convert to float for consistency)
            pos_key = (float(x), float(y))
            links = np.random.binomial(1, los_probability, num_beacons)
            los_map[pos_key] = links.tolist()
    
    return los_map


def discretize_position(x: float, y: float, grid_resolution: float = 1.0) -> Tuple[float, float]:
    """
    Convert continuous position to discretized grid position.
    
    Args:
        x: X coordinate
        y: Y coordinate
        grid_resolution: Resolution of the grid
    
    Returns:
        Tuple of (x, y) discretized to grid resolution
    """
    discretized_x = round(x / grid_resolution) * grid_resolution
    discretized_y = round(y / grid_resolution) * grid_resolution
    return (discretized_x, discretized_y)


def save_los_map(los_map: Dict[Tuple[int, int], List[int]], filename: str = None) -> str:
    """
    Save LoS map to a JSON file.
    
    Args:
        los_map: Dictionary of LoS links
        filename: Path to save file (if None, uses timestamp)
    
    Returns:
        Path to saved file
    """
    if filename is None:
        from datetime import datetime
        los_dir = Path(__file__).parent.parent / 'los_maps'
        los_dir.mkdir(exist_ok=True)
        filename = str(los_dir / f"los_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert tuple keys to strings for JSON serialization
    los_map_serializable = {str(k): v for k, v in los_map.items()}
    
    with open(filename, 'w') as f:
        json.dump(los_map_serializable, f, indent=2)
    
    print(f"LoS map saved to {filename}")
    return filename


def load_los_map(filename: str) -> Dict[Tuple[float, float], List[int]]:
    """
    Load LoS map from a JSON file.
    
    Args:
        filename: Path to JSON file
    
    Returns:
        Dictionary of LoS links with Tuple(float, float) keys
    """
    with open(filename, 'r') as f:
        los_map_serializable = json.load(f)
    
    # Convert string keys back to tuples of floats
    los_map = {}
    for k_str, v in los_map_serializable.items():
        # Parse string like "(10.5, 20.3)" to tuple (10.5, 20.3)
        k_str = k_str.strip()
        if k_str.startswith('(') and k_str.endswith(')'):
            k_str = k_str[1:-1]  # Remove parentheses
        coords = [float(coord.strip()) for coord in k_str.split(',')]
        los_map[tuple(coords)] = v
    
    print(f"LoS map loaded from {filename}")
    return los_map
