"""
Generate and save LoS link maps for use in training and evaluation.

This script generates pre-computed LoS/NLoS link patterns for all grid positions
and saves them to a file. These saved maps are then reused across multiple training
episodes and evaluations for consistency and efficiency.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from core.environment import Environment
from core.link_model import generate_los_map, save_los_map
from config import (
    GRID_SIZE, NUM_BEACONS, LOS_PROBABILITY, AGENT_STEP_SIZE
)


def generate_and_save_los_maps(num_scenarios: int = 1):
    """
    Generate and save LoS maps for training and evaluation.
    
    Args:
        num_scenarios: Number of different LoS map scenarios to generate
                      (useful for testing robustness with different RF environments)
    
    Returns:
        List of saved file paths
    """
    saved_paths = []
    
    print("=" * 70)
    print("GENERATING AND SAVING LOS LINK MAPS")
    print("=" * 70 + "\n")
    
    print(f"Configuration:")
    print(f"  Grid Size: {GRID_SIZE} x {GRID_SIZE}")
    print(f"  Number of Beacons: {NUM_BEACONS}")
    print(f"  LoS Probability: {LOS_PROBABILITY}")
    print(f"  Agent Step Size: {AGENT_STEP_SIZE}\n")
    
    los_dir = Path(__file__).parent / 'los_maps'
    los_dir.mkdir(exist_ok=True)
    
    for scenario_idx in range(num_scenarios):
        print(f"Generating scenario {scenario_idx + 1}/{num_scenarios}...")
        
        # Generate LoS map
        los_map = generate_los_map(
            grid_width=int(GRID_SIZE),
            grid_height=int(GRID_SIZE),
            num_beacons=NUM_BEACONS,
            los_probability=LOS_PROBABILITY,
            grid_resolution=AGENT_STEP_SIZE,
        )
        
        print(f"  Generated {len(los_map)} grid positions")
        
        # Save with descriptive name
        filename = los_dir / f"los_map_scenario_{scenario_idx + 1}.json"
        
        # Convert tuple keys to strings for JSON serialization
        los_map_serializable = {str(k): v for k, v in los_map.items()}
        
        with open(filename, 'w') as f:
            json.dump(los_map_serializable, f, indent=2)
        
        print(f"  Saved to: {filename}\n")
        saved_paths.append(str(filename))
    
    print("=" * 70)
    print(f"SUCCESS: Generated and saved {num_scenarios} LoS map(s)")
    print("=" * 70 + "\n")
    
    return saved_paths


def get_default_los_map_path():
    """
    Get the path to the default LoS map file.
    
    Returns:
        Path to default LoS map (scenario 1) or None if not found
    """
    default_map = Path(__file__).parent / 'los_maps' / 'los_map_scenario_1.json'
    
    if default_map.exists():
        return str(default_map)
    else:
        return None


def verify_los_map(los_map_path: str):
    """
    Verify that a LoS map file is valid and print statistics.
    
    Args:
        los_map_path: Path to LoS map JSON file
    """
    print("=" * 70)
    print("VERIFYING LOS MAP")
    print("=" * 70 + "\n")
    
    if not Path(los_map_path).exists():
        print(f"✗ File not found: {los_map_path}")
        return False
    
    try:
        with open(los_map_path, 'r') as f:
            los_map_serializable = json.load(f)
        
        # Convert back to tuple keys
        los_map = {tuple(map(int, k.strip('()').split(','))): v 
                   for k, v in los_map_serializable.items()}
        
        # Verify structure
        num_positions = len(los_map)
        num_beacons = len(list(los_map.values())[0]) if los_map else 0
        
        print(f"✓ File valid: {los_map_path}")
        print(f"  Grid positions: {num_positions}")
        print(f"  Beacons per position: {num_beacons}")
        
        # Calculate statistics
        all_los_counts = []
        for links in los_map.values():
            los_count = sum(links)
            all_los_counts.append(los_count)
        
        import numpy as np
        avg_los = np.mean(all_los_counts)
        print(f"  Average LoS beacons per position: {avg_los:.2f}/{num_beacons}")
        print(f"  Min LoS beacons: {min(all_los_counts)}")
        print(f"  Max LoS beacons: {max(all_los_counts)}")
        
        print("\n✓ LoS map verification passed!")
        print("=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying LoS map: {e}")
        print("=" * 70 + "\n")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate and save LoS link maps for training and evaluation'
    )
    parser.add_argument(
        '--num-scenarios',
        type=int,
        default=1,
        help='Number of LoS map scenarios to generate (default: 1)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the generated LoS map file'
    )
    
    args = parser.parse_args()
    
    # Generate LoS maps
    saved_paths = generate_and_save_los_maps(num_scenarios=args.num_scenarios)
    
    # Verify if requested
    if args.verify and saved_paths:
        verify_los_map(saved_paths[0])
