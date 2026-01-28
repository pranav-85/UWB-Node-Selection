"""
Sample simulation demonstrating beacon selection until battery depletion.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from rl.trainer_dqn import DQNTrainer
from rl.trainer_ppo import PPOActorCritic
from config import NUM_BEACONS, NUM_SELECTED_BEACONS


class SimulationRecorder:
    """Record simulation data."""
    
    def __init__(self):
        self.timesteps = []
        self.agent_positions = []
        self.selected_beacons_list = []
        self.battery_levels_history = {i: [] for i in range(NUM_BEACONS)}
        self.rewards = []
        self.errors = []
        self.los_selections = {i: 0 for i in range(NUM_BEACONS)}
    
    def record_step(self, timestep, agent_pos, selected_beacons, battery_levels, 
                   los_flags, reward, error):
        """Record a single simulation step."""
        self.timesteps.append(timestep)
        self.agent_positions.append(agent_pos)
        self.selected_beacons_list.append(selected_beacons)
        
        for i in range(NUM_BEACONS):
            self.battery_levels_history[i].append(battery_levels[i])
        
        self.rewards.append(reward)
        self.errors.append(error)
        
        # Safely track LoS selections with bounds checking
        for idx in selected_beacons:
            if 0 <= idx < len(los_flags) and 0 <= idx < NUM_BEACONS:
                if los_flags[idx]:
                    self.los_selections[idx] += 1


def run_sample_simulation(method='dqn', visualize=True, los_map_file=None):
    """
    Run a sample simulation until battery depletion.
    
    Args:
        method: 'dqn', 'ppo', 'random', or 'nearest_neighbor'
        visualize: Whether to show visualizations
        los_map_file: Path to pre-computed LoS map
    
    Returns:
        SimulationRecorder with results
    """
    
    print("\n" + "="*70)
    print(f"SAMPLE SIMULATION - {method.upper()}")
    print("="*70 + "\n")
    
    # Initialize environment
    env = Environment(los_map_file=los_map_file)
    env.reset_agent_to_random_location()
    env.reset_beacon_batteries()
    
    recorder = SimulationRecorder()
    timestep = 0
    
    # Load trained model if using RL methods
    if method == 'dqn':
        print("Loading DQN model...")
        state_size = 2 + NUM_BEACONS + NUM_BEACONS
        trainer = DQNTrainer(state_size=state_size)
        dqn_model_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
        if dqn_model_path.exists():
            trainer.load_model(str(dqn_model_path))
            trainer.epsilon = 0.0
        else:
            print(f"Warning: Model not found at {dqn_model_path}")
    
    elif method == 'ppo':
        print("Loading PPO model...")
        state_size = 2 + NUM_BEACONS + NUM_BEACONS
        action_dim = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))
        ppo_model = PPOActorCritic(state_dim=state_size, action_dim=action_dim)
        ppo_model_path = Path(__file__).parent.parent / 'models' / 'ppo_model.pt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if ppo_model_path.exists():
            ppo_model.load_state_dict(torch.load(str(ppo_model_path), map_location=device))
            ppo_model.to(device)
            ppo_model.eval()
        else:
            print(f"Warning: Model not found at {ppo_model_path}")
    
    print(f"Agent starting position: {env.agent.get_position()}")
    print(f"Initial battery levels: {[f'{b:.1f}%' for b in env.get_battery_levels()]}\n")
    
    # Create figure for live visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Run simulation until battery depletion
    battery_depleted = False
    max_steps = 500
    
    while not battery_depleted and timestep < max_steps:
        timestep += 1
        env.step()
        
        # Select beacons based on method
        if method == 'dqn':
            state = trainer.state_to_vector(env)
            action = trainer.select_action(state, training=False)
            selected_indices = list(trainer.possible_actions[action])
        
        elif method == 'ppo':
            state = np.array(list(env.agent.get_position()) + env.get_battery_levels() + 
                           (env.current_links if env.current_links is not None else [0] * NUM_BEACONS),
                           dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = ppo_model(state_tensor)
            action_probs = torch.softmax(logits, dim=1)
            action = action_probs.argmax(dim=1).item()
            possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
            selected_indices = list(possible_actions[action])
        
        elif method == 'random':
            selected_indices = list(np.random.choice(NUM_BEACONS, NUM_SELECTED_BEACONS, replace=False))
        
        elif method == 'nearest_neighbor':
            agent_pos = np.array(env.agent.get_position())
            beacon_positions = np.array([beacon.position for beacon in env.beacons])
            distances = np.linalg.norm(beacon_positions - agent_pos, axis=1)
            selected_indices = list(np.argsort(distances)[:NUM_SELECTED_BEACONS])
        
        env.selected_beacon_indices = selected_indices
        
        # Calculate metrics
        agent_pos = np.array(env.agent.get_position())
        selected_positions = np.array([env.beacons[i].position for i in selected_indices])
        centroid = np.mean(selected_positions, axis=0)
        error = np.linalg.norm(agent_pos - centroid)
        
        los_flags = [env.current_links[i] if i < len(env.current_links) else 0 for i in range(NUM_BEACONS)]
        battery_levels = env.get_battery_levels()
        reward = compute_reward(agent_pos, selected_positions, los_flags, battery_levels)
        
        # Record
        recorder.record_step(timestep, agent_pos, selected_indices, battery_levels,
                           los_flags, reward, error)
        
        # Visualize every 5 steps
        if timestep % 5 == 0 or timestep == 1:
            fig, ax = env.visualize(
                title=f"{method.upper()} Simulation - Timestep {timestep} | "
                      f"Reward: {reward:.3f} | Min Battery: {min(battery_levels):.1f}%",
                ax=ax,
                figsize=(10, 10)
            )
            plt.pause(0.5)
            plt.draw()
        
        # Check for depletion
        if any(level <= 0 for level in battery_levels):
            battery_depleted = True
            
            # Final visualization
            fig, ax = env.visualize(
                title=f"{method.upper()} Simulation - COMPLETED at Timestep {timestep} | "
                      f"*** BATTERY DEPLETED ***",
                ax=ax,
                figsize=(10, 10)
            )
            plt.pause(2)
            print(f"\n*** Battery Depleted at timestep {timestep} ***\n")
    
    plt.close(fig)
    return recorder, env, timestep


def plot_simulation_results(recorder, env, timestep, method):
    """Generate summary of simulation results."""
    
    summary = f"""
    
{'='*70}
SIMULATION SUMMARY - {method.upper()}
{'='*70}

Duration: {timestep} timesteps
Final Agent Position: ({recorder.agent_positions[-1][0]:.2f}, {recorder.agent_positions[-1][1]:.2f})

Final Battery Levels:
"""
    for i in range(NUM_BEACONS):
        final_battery = recorder.battery_levels_history[i][-1]
        summary += f"  Beacon {i}: {final_battery:>6.2f}%\n"
    
    summary += f"\nReward Metrics:\n"
    summary += f"  Total Reward: {sum(recorder.rewards):.2f}\n"
    summary += f"  Avg Reward/step: {np.mean(recorder.rewards):.4f}\n"
    summary += f"  Mean Localization Error: {np.mean(recorder.errors):.4f} m\n"
    summary += f"\n{'='*70}\n"
    
    return summary


def main():
    # Run simulations with different methods
    methods = ['dqn', 'ppo', 'random', 'nearest_neighbor']
    results = {}
    
    # Use pre-computed LoS map if available
    from generate_links import get_default_los_map_path
    los_map_file = get_default_los_map_path()
    
    print(f"Using LoS map: {los_map_file if los_map_file else 'Generated new map'}\n")
    
    for method in methods:
        try:
            recorder, env, timestep = run_sample_simulation(method=method, los_map_file=los_map_file)
            results[method] = {
                'recorder': recorder,
                'env': env,
                'timestep': timestep,
                'total_reward': sum(recorder.rewards),
                'mean_error': np.mean(recorder.errors),
                'final_batteries': [recorder.battery_levels_history[i][-1] for i in range(NUM_BEACONS)]
            }
            
            summary = plot_simulation_results(recorder, env, timestep, method)
            print(summary)
            
        except Exception as e:
            print(f"\nError running {method} simulation: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Method':<20} {'Duration':<12} {'Total Reward':<15} {'Mean Error':<12} {'Min Battery':<12}")
    print("-" * 70)
    
    for method in methods:
        if method in results:
            r = results[method]
            min_battery = min(r['final_batteries'])
            print(f"{method:<20} {r['timestep']:<12} {r['total_reward']:<15.2f} "
                  f"{r['mean_error']:<12.4f} {min_battery:<12.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
