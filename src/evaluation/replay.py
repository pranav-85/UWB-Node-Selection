import sys
from pathlib import Path
import argparse
import csv
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.environment import Environment
from config import VISUALIZATION_FIGSIZE, STEP_PAUSE_TIME


def load_scenario(filename: str):
    """
    Load scenario from CSV file.
    
    Args:
        filename: Scenario filename (with or without .csv extension)
    
    Returns:
        List of dictionaries containing timestep data
    """
    scenarios_dir = Path(__file__).parent.parent / 'scenarios'
    
    # Add .csv extension if not present
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    
    filepath = scenarios_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scenario file not found: {filepath}")
    
    records = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert string values to appropriate types
            record = {
                'timestep': int(row['timestep']),
                'agent_x': float(row['agent_x']),
                'agent_y': float(row['agent_y']),
                'selected_beacons_rns': [int(x) for x in row['selected_beacons_rns'].split(',')],
                'los_links': [int(x) for x in row['los_links'].split(',')]
            }
            records.append(record)
    
    return records


def replay_scenario(scenario_records):
    """
    Replay a recorded scenario.
    
    Args:
        scenario_records: List of timestep records from loaded scenario
    """
    plt.ion()
    env = Environment()
    
    # Create figure
    fig, ax = plt.subplots(figsize=VISUALIZATION_FIGSIZE)
    
    for record in scenario_records:
        # Set agent position
        env.agent.x = record['agent_x']
        env.agent.y = record['agent_y']
        
        # Set selected beacons
        env.selected_beacon_indices = record['selected_beacons_rns']
        
        # Set LoS links
        env.current_links = record['los_links']
        
        # Visualize
        ax.clear()
        env.visualize(title=f"Replay - Step {record['timestep']}", ax=ax)
        
        # Update display
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        plt.pause(STEP_PAUSE_TIME)
    
    # Keep final frame visible
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Replay a recorded UWB simulation scenario'
    )
    parser.add_argument(
        'scenario',
        type=str,
        help='Scenario filename (with or without .csv extension)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Loading scenario: {args.scenario}")
        records = load_scenario(args.scenario)
        print(f"Loaded {len(records)} timesteps")
        
        print("Starting replay...")
        replay_scenario(records)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during replay: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
