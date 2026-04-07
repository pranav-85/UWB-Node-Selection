"""
Domain Generalization Training for DQN-based UWB Anchor Selection

Trains a single DQN model across multiple randomized environments.
Each environment has different grid size, beacon count, and CIR parameters.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import torch
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from trainer_dqn import DQNTrainer
from cir_training_config import setup_cir_training, FAST_TRAINING
from config import NUM_BEACONS, NUM_SELECTED_BEACONS
from reward.reward import compute_reward


# Configuration constants
MIN_GRID = 10
MAX_GRID = 30
MIN_BEACONS = 6
MAX_BEACONS = 12

MIN_LOS_PROB = 0.3
MAX_LOS_PROB = 0.8

MIN_CLUSTERS = 2
MAX_CLUSTERS = 4
MIN_RAYS = 3
MAX_RAYS = 5

MIN_LOS_STD = 0.03
MAX_LOS_STD = 0.08

MIN_NLOS_BIAS = 0.3
MAX_NLOS_BIAS = 1.0

MIN_BATTERY = 80.0
MAX_BATTERY = 120.0
MIN_CONSUMPTION = 2.0
MAX_CONSUMPTION = 4.0


def generate_environment_configs(num_envs: int, save_path: str) -> List[Dict[str, Any]]:
    """
    Generate randomized environment configurations for domain generalization.
    
    Args:
        num_envs: Number of environment configs to generate
        save_path: Path to save configs as JSON
    
    Returns:
        List of environment configuration dictionaries
    """
    configs = []
    
    for env_id in range(num_envs):
        # Randomize grid dimensions
        grid_width = random.randint(MIN_GRID, MAX_GRID)
        grid_height = random.randint(MIN_GRID, MAX_GRID)
        
        # Randomize number of beacons
        num_beacons = random.randint(MIN_BEACONS, MAX_BEACONS)
        
        # Generate random beacon positions (avoid clustering)
        beacon_positions = []
        min_distance = 2.0
        max_attempts = 100
        
        for _ in range(num_beacons):
            for _ in range(max_attempts):
                x = random.uniform(1.0, grid_width - 1.0)
                y = random.uniform(1.0, grid_height - 1.0)
                
                # Check distance from existing beacons
                valid = True
                for bx, by in beacon_positions:
                    if np.sqrt((x - bx)**2 + (y - by)**2) < min_distance:
                        valid = False
                        break
                
                if valid:
                    beacon_positions.append([x, y])
                    break
        
        # Ensure minimum beacons after filtering
        if len(beacon_positions) < 6:
            beacon_positions = [[random.uniform(1, grid_width-1), 
                                random.uniform(1, grid_height-1)] 
                               for _ in range(6)]
        
        beacon_positions = beacon_positions[:num_beacons]
        
        # LoS probability
        los_probability = random.uniform(MIN_LOS_PROB, MAX_LOS_PROB)
        
        # CIR parameters
        num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
        rays_per_cluster = random.randint(MIN_RAYS, MAX_RAYS)
        los_max_clusters = random.randint(MIN_CLUSTERS, min(MAX_CLUSTERS, num_clusters))
        delay_spread = random.uniform(20, 200)  # nanoseconds
        decay_factor = random.uniform(0.8, 1.2)
        
        # Noise parameters
        los_std = random.uniform(MIN_LOS_STD, MAX_LOS_STD)
        nlos_bias_min = random.uniform(MIN_NLOS_BIAS, 0.5)
        nlos_bias_max = random.uniform(nlos_bias_min, MAX_NLOS_BIAS)
        
        # Battery parameters
        initial_battery = random.uniform(MIN_BATTERY, MAX_BATTERY)
        consumption_multiplier = random.uniform(MIN_CONSUMPTION, MAX_CONSUMPTION)
        
        config = {
            'env_id': env_id,
            'grid_width': grid_width,
            'grid_height': grid_height,
            'num_beacons': num_beacons,
            'beacon_positions': beacon_positions,
            'los_probability': los_probability,
            'cir': {
                'num_clusters': num_clusters,
                'rays_per_cluster': rays_per_cluster,
                'los_max_clusters': los_max_clusters,
                'delay_spread_ns': delay_spread,
                'decay_factor': decay_factor,
            },
            'noise': {
                'los_std': los_std,
                'nlos_bias_min': nlos_bias_min,
                'nlos_bias_max': nlos_bias_max,
            },
            'battery': {
                'initial_level': initial_battery,
                'consumption_multiplier': consumption_multiplier,
            }
        }
        
        configs.append(config)
    
    # Save configs
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {num_envs} environment configs -> {save_path}")
    return configs


def load_environment_configs(filepath: str) -> List[Dict[str, Any]]:
    """
    Load environment configurations from JSON file.
    
    Args:
        filepath: Path to JSON config file
    
    Returns:
        List of configuration dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        configs = json.load(f)
    
    print(f"Loaded {len(configs)} environment configs from {filepath}")
    return configs


def create_env_from_config(config: Dict[str, Any]) -> Environment:
    """
    Create an Environment instance from a configuration dictionary.
    
    Args:
        config: Environment configuration dictionary
    
    Returns:
        Initialized Environment instance
    """
    # Create environment with grid size
    env = Environment(
        grid_size=max(config['grid_width'], config['grid_height']),
        los_map=None,
        los_map_file=None
    )
    
    # Override beacon positions from config
    for i, pos in enumerate(config['beacon_positions'][:len(env.beacons)]):
        if i < len(env.beacons):
            env.beacons[i].position = np.array(pos, dtype=float)
    
    # Store config metadata
    env.config = config
    env.env_id = config['env_id']
    
    return env


def train_across_environments(
    configs: List[Dict[str, Any]],
    trainer: DQNTrainer,
    num_episodes: int,
    max_steps: int = 2000
) -> Dict[str, Any]:
    """
    Train DQN across multiple randomized environments (domain generalization).
    
    Args:
        configs: List of environment configurations
        trainer: DQNTrainer instance (shared across environments)
        num_episodes: Total number of training episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Training statistics dictionary
    """
    # Setup CIR
    setup_cir_training(FAST_TRAINING)
    
    episode_rewards = []
    episode_lengths = []
    episode_envs = []
    episode_beacons = []
    episode_grid_sizes = []
    
    losses = []
    
    pbar = tqdm(range(num_episodes), desc='Domain Generalization Training', position=0)
    
    for episode in pbar:
        # Randomly sample environment config
        config = random.choice(configs)
        env = create_env_from_config(config)
        env_id = config['env_id']
        num_beacons = len(config['beacon_positions'])
        grid_size = max(config['grid_width'], config['grid_height'])
        
        # Reset environment
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()
        
        state = trainer.state_to_vector(env)
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action
            action = trainer.select_action(state, training=True)
            selected_beacons = list(trainer.possible_actions[action])
            
            # Apply action
            env.selected_beacon_indices = selected_beacons
            env.step()
            
            # Get environment state
            agent_pos = env.agent.get_position()
            beacon_positions = [env.beacons[i].position for i in selected_beacons]
            los_flags = [env.current_links[i] for i in selected_beacons]
            battery_levels = env.get_battery_levels()
            
            # Compute reward
            reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
            reward = np.clip(reward, -1.0, 1.0)
            
            # Get next state
            next_state = trainer.state_to_vector(env)
            
            # Check termination
            min_battery = min(battery_levels)
            done = (min_battery <= 10) or (step == max_steps - 1)
            
            # Store in replay buffer
            trainer.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train
            if len(trainer.replay_buffer) >= trainer.warmup_buffer_size:
                loss = trainer.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Decay epsilon
            trainer.epsilon = max(trainer.epsilon_end, trainer.epsilon * trainer.epsilon_decay)
            
            if done:
                break
        
        # Update target network periodically
        if (episode + 1) % 20 == 0:
            trainer.update_target_network()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_envs.append(env_id)
        episode_beacons.append(num_beacons)
        episode_grid_sizes.append(grid_size)
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards[-10:])
        avg_loss = np.mean(losses[-10:]) if losses else 0
        
        pbar.set_postfix({
            'Avg Reward': f'{avg_reward:.4f}',
            'Avg Loss': f'{avg_loss:.6f}',
            'Epsilon': f'{trainer.epsilon:.4f}',
            'Buffer': len(trainer.replay_buffer),
            'Env ID': env_id,
            'Beacons': num_beacons
        })
    
    pbar.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_envs': episode_envs,
        'episode_beacons': episode_beacons,
        'episode_grid_sizes': episode_grid_sizes,
        'losses': losses,
    }


def evaluate_generalization(
    trainer: DQNTrainer,
    num_test_envs: int,
    max_steps: int = 2000
) -> Dict[str, Any]:
    """
    Evaluate trained model on unseen randomized environments.
    
    Args:
        trainer: Trained DQNTrainer instance
        num_test_envs: Number of test environments to generate
        max_steps: Maximum steps per episode
    
    Returns:
        Evaluation statistics dictionary
    """
    print("\nGenerating unseen test environments...")
    test_configs = generate_environment_configs(num_test_envs, 'data/test_configs.json')
    
    localization_errors = []
    rewards = []
    episode_lengths = []
    
    print("Evaluating on unseen environments...")
    
    pbar = tqdm(test_configs, desc='Evaluating Generalization', position=0)
    
    for config in pbar:
        env = create_env_from_config(config)
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()
        
        state = trainer.state_to_vector(env)
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = trainer.select_action(state, training=False)
            selected_beacons = list(trainer.possible_actions[action])
            
            # Apply action
            env.selected_beacon_indices = selected_beacons
            env.step()
            
            # Get state
            agent_pos = env.agent.get_position()
            beacon_positions = [env.beacons[i].position for i in selected_beacons]
            los_flags = [env.current_links[i] for i in selected_beacons]
            battery_levels = env.get_battery_levels()
            
            # Compute reward
            reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
            
            # Next state
            next_state = trainer.state_to_vector(env)
            
            # Check termination
            min_battery = min(battery_levels)
            done = (min_battery <= 10) or (step == max_steps - 1)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    pbar.close()
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'all_rewards': rewards,
        'all_lengths': episode_lengths,
    }


def main():
    """Main function for domain generalization training."""
    print("\n" + "="*70)
    print("DOMAIN GENERALIZATION TRAINING FOR DQN-BASED UWB ANCHOR SELECTION")
    print("="*70 + "\n")
    
    # Paths
    config_dir = Path('data/domain_generalization')
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path('checkpoints/domain_generalization')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training environment configs
    num_train_envs = 40
    config_path = config_dir / f'train_configs_{num_train_envs}.json'
    
    if config_path.exists():
        print(f"Loading existing configs from {config_path}")
        train_configs = load_environment_configs(str(config_path))
    else:
        print(f"Generating {num_train_envs} training environment configs...")
        train_configs = generate_environment_configs(num_train_envs, str(config_path))
    
    # Print config summary
    print("\nTraining Environment Summary:")
    grid_sizes = [max(c['grid_width'], c['grid_height']) for c in train_configs]
    beacon_counts = [len(c['beacon_positions']) for c in train_configs]
    los_probs = [c['los_probability'] for c in train_configs]
    
    print(f"  Grid sizes: {min(grid_sizes):.0f}-{max(grid_sizes):.0f} (avg: {np.mean(grid_sizes):.0f})")
    print(f"  Beacon counts: {min(beacon_counts)}-{max(beacon_counts)} (avg: {np.mean(beacon_counts):.1f})")
    print(f"  LoS probabilities: {min(los_probs):.2f}-{max(los_probs):.2f} (avg: {np.mean(los_probs):.2f})")
    print()
    
    # Initialize trainer
    print("Initializing DQN trainer...")
    state_size = NUM_BEACONS  # Fixed state size
    
    trainer = DQNTrainer(
        state_size=state_size,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        buffer_capacity=10000,
        batch_size=32,
        warmup_buffer_size=1000
    )
    
    print(f"  State size: {state_size}")
    print(f"  Action size: {trainer.action_size}")
    print(f"  Device: {trainer.device}\n")
    
    # Training
    print("="*70)
    print("PHASE 1: DOMAIN GENERALIZATION TRAINING")
    print("="*70 + "\n")
    
    num_episodes = 400
    training_stats = train_across_environments(
        train_configs,
        trainer,
        num_episodes=num_episodes,
        max_steps=2000
    )
    
    print(f"\nTraining completed!")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Final average reward: {np.mean(training_stats['episode_rewards'][-10:]):.4f}")
    print(f"  Final epsilon: {trainer.epsilon:.4f}\n")
    
    # Save model
    print("="*70)
    print("PHASE 2: EVALUATION ON UNSEEN ENVIRONMENTS")
    print("="*70 + "\n")
    
    # Evaluate on new environments
    num_test_envs = 10
    eval_stats = evaluate_generalization(trainer, num_test_envs, max_steps=2000)
    
    print(f"\nEvaluation Results on {num_test_envs} Unseen Environments:")
    print(f"  Mean reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
    print(f"  Mean episode length: {eval_stats['mean_length']:.0f} ± {eval_stats['std_length']:.0f}")
    print()
    
    # Save trained model
    model_path = checkpoint_dir / f'dqn_domain_generalization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    trainer.save_model(str(model_path))
    
    # Save training log
    log_path = checkpoint_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'training': {
            'num_episodes': num_episodes,
            'num_train_envs': num_train_envs,
            'final_avg_reward': float(np.mean(training_stats['episode_rewards'][-10:])),
            'total_transitions': len(trainer.replay_buffer),
        },
        'evaluation': {
            'num_test_envs': num_test_envs,
            'mean_reward': float(eval_stats['mean_reward']),
            'std_reward': float(eval_stats['std_reward']),
            'mean_length': float(eval_stats['mean_length']),
        },
        'model_path': str(model_path),
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Log saved to: {log_path}")
    print("\n" + "="*70)
    print("Domain generalization training complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
