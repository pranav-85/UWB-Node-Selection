import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from localization.trilateration import trilateration_2d, compute_noisy_distances
from config import NUM_BEACONS, NUM_SELECTED_BEACONS

def select_nearest_beacons(agent_pos, beacons, k):
    """
    Select k nearest beacons to the agent.
    
    Args:
        agent_pos: np.array([x, y])
        beacons: list of Beacon objects
        k: number of beacons to select
    Returns:
        selected_indices: list of beacon indices
    """
    distances = []

    for i, beacon in enumerate(beacons):
        bx, by = beacon.position
        d = np.linalg.norm(agent_pos - np.array([bx, by]))
        distances.append((i, d))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Select k nearest
    selected_indices = [idx for idx, _ in distances[:k]]

    return selected_indices


def run_visualization():
    print("Initializing environment...")
    env = Environment()
    
    # Critical battery threshold (10%)
    CRITICAL_BATTERY_THRESHOLD = 10.0
    
    # Setup visualization
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))
    
    print("Starting simulation with Random Beacon Selection...")
    print("Press Ctrl+C to stop manually.")
    
    step = 0
    while True:
        # 1. Random Selection
        agent_pos = np.array(env.agent.get_position())

        selected_indices = select_nearest_beacons(
            agent_pos=agent_pos,
            beacons=env.beacons,
            k=NUM_SELECTED_BEACONS
        )

        env.selected_beacon_indices = selected_indices

        
        # 2. Step Environment
        env.step()
        
        # 3. Calculate Error
        agent_pos = np.array(env.agent.get_position())
        selected_positions = np.array([env.beacons[i].position for i in selected_indices])
        los_flags = [env.current_links[i] for i in selected_indices]
        
        distances = compute_noisy_distances(agent_pos, selected_positions, los_flags)
        est_x, est_y = trilateration_2d(selected_positions, distances)
        est_pos = np.array([est_x, est_y])
        
        error = np.sqrt(np.sum((agent_pos - est_pos) ** 2))
        
        # 4. Check Battery
        battery_levels = env.get_battery_levels()
        min_battery = min(battery_levels)
        
        # 5. Visualize
        env.visualize(title=f"Step: {step} | Random Selection | Min Battery: {min_battery:.1f}%", ax=ax)
        
        # Add Error Text to Plot
        ax.text(0.02, 0.98, f"Localization Error: {error:.4f} m", 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.draw()
        plt.pause(0.5)  # Pause to show frame
        
        if min_battery <= CRITICAL_BATTERY_THRESHOLD:
            print(f"\nSimulation ended at step {step}: Battery below threshold ({min_battery:.1f}%)")
            plt.show(block=True)
            break
            
        step += 1

if __name__ == "__main__":
    try:
        run_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
