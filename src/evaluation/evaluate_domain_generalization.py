import sys
from pathlib import Path
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from rl.trainer_dqn import DQNTrainer
from localization.trilateration import trilateration_2d, compute_noisy_distances, localization_error
from config import NUM_BEACONS, NUM_SELECTED_BEACONS


class DomainGeneralizationMetrics:
    """Collect and compute domain generalization evaluation metrics."""
    
    def __init__(self):
        self.results_by_config = defaultdict(list)  # config_id -> list of metrics
        self.results_by_grid_size = defaultdict(list)
        self.results_by_beacon_count = defaultdict(list)
        self.results_by_los_prob = defaultdict(list)
        self.all_errors = []
        self.all_rewards = []
        self.trajectory_lengths = []
    
    def add_episode_result(self, config_id: int, config: dict, 
                          localization_error: float, reward: float, 
                          trajectory_length: int):
        """Record results for an episode."""
        self.results_by_config[config_id].append({
            'error': localization_error,
            'reward': reward,
            'trajectory_length': trajectory_length
        })
        
        self.results_by_grid_size[config['grid_width']].append(localization_error)
        self.results_by_beacon_count[config['num_beacons']].append(localization_error)
        self.results_by_los_prob[round(config['los_probability'], 1)].append(localization_error)
        
        self.all_errors.append(localization_error)
        self.all_rewards.append(reward)
        self.trajectory_lengths.append(trajectory_length)
    
    def get_summary_metrics(self):
        """Get summary statistics across all evaluations."""
        return {
            'mean_error': np.mean(self.all_errors),
            'std_error': np.std(self.all_errors),
            'median_error': np.median(self.all_errors),
            'min_error': np.min(self.all_errors),
            'max_error': np.max(self.all_errors),
            'error_90th': np.percentile(self.all_errors, 90),
            'error_95th': np.percentile(self.all_errors, 95),
            'mean_reward': np.mean(self.all_rewards),
            'mean_trajectory_length': np.mean(self.trajectory_lengths),
        }
    
    def get_metrics_by_environment_param(self, param_name: str):
        """Get performance metrics grouped by environment parameter."""
        if param_name == 'grid_size':
            data = self.results_by_grid_size
        elif param_name == 'beacon_count':
            data = self.results_by_beacon_count
        elif param_name == 'los_prob':
            data = self.results_by_los_prob
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        result = {}
        for key, errors in data.items():
            if errors:
                result[key] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'count': len(errors),
                }
        return result


def generate_evaluation_configs(num_configs: int = 50, seed: int = 999):
    """
    Generate evaluation configurations (different from training configs).
    Creates diverse environment configurations for testing generalization.
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    configs = []
    
    for i in range(num_configs):
        grid_width = np.random.randint(12, 28)
        grid_height = np.random.randint(12, 28)
        num_beacons = np.random.randint(7, 11)
        
        # Generate beacon positions
        beacon_positions = []
        min_distance = 2.0
        max_attempts = 50
        
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
        
        # Ensure minimum beacons
        if len(beacon_positions) < 6:
            beacon_positions = [[random.uniform(1, grid_width-1), 
                                random.uniform(1, grid_height-1)] 
                               for _ in range(6)]
        
        beacon_positions = beacon_positions[:num_beacons]
        
        config = {
            'config_id': i,
            'env_id': f'eval_{i}',
            'grid_width': grid_width,
            'grid_height': grid_height,
            'num_beacons': num_beacons,
            'beacon_positions': beacon_positions,
            'los_probability': np.random.uniform(0.25, 0.85),
            'cir_clusters': np.random.randint(2, 5),
            'cir_rays_per_cluster': np.random.randint(3, 6),
            'noise_std': np.random.uniform(0.1, 0.5),
        }
        configs.append(config)
    
    return configs


def create_env_from_config(config: dict) -> Environment:
    """Create an Environment instance from a configuration dictionary."""
    # Create environment with max of grid dimensions
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
    env.env_id = config.get('env_id', f"config_{config['config_id']}")
    
    return env


def evaluate_on_config(trainer: DQNTrainer, config: dict, num_episodes: int = 10,
                      max_steps_per_episode: int = 100) -> dict:
    """
    Evaluate the domain-generalized model on a specific configuration.
    
    Returns:
        dict with metrics for this configuration
    """
    metrics = {
        'config_id': config['config_id'],
        'config': config,
        'episode_errors': [],
        'episode_rewards': [],
        'episode_lengths': [],
    }
    
    for episode in range(num_episodes):
        env = create_env_from_config(config)
        
        state = trainer.state_to_vector(env)
        episode_error = 0.0
        episode_reward = 0.0
        steps = 0
        
        for step in range(max_steps_per_episode):
            # Select action using trained policy
            action = trainer.select_action(state, training=False)
            selected_indices = list(trainer.possible_actions[action])
            
            # Step environment
            env.selected_beacon_indices = selected_indices
            env.step()
            
            # Get state and reward
            agent_pos = np.array(env.agent.get_position())
            selected_positions = np.array([env.beacons[i].position for i in selected_indices])
            los_flags = [env.current_links[i] for i in selected_indices]
            battery_levels = env.get_battery_levels()
            
            # Compute localization error
            distances = compute_noisy_distances(agent_pos, selected_positions, los_flags)
            est_x, est_y = trilateration_2d(selected_positions, distances)
            est_pos = np.array([est_x, est_y])
            error = np.sqrt(np.sum((agent_pos - est_pos) ** 2))
            
            # Compute reward
            reward = compute_reward(agent_pos, selected_positions, los_flags, battery_levels)
            
            episode_error += error
            episode_reward += reward
            steps += 1
            
            # Update state
            state = trainer.state_to_vector(env)
            
            # Early termination if battery critical
            if min(battery_levels) <= 10:
                break
        
        # Normalize metrics
        avg_error = episode_error / steps if steps > 0 else 0.0
        metrics['episode_errors'].append(avg_error)
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
    
    # Compute summary statistics for this config
    metrics['mean_error'] = np.mean(metrics['episode_errors'])
    metrics['std_error'] = np.std(metrics['episode_errors'])
    metrics['mean_reward'] = np.mean(metrics['episode_rewards'])
    
    return metrics


def plot_generalization_heatmap(all_results: list):
    """Plot heatmap of error vs (grid size, beacon count)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract grid sizes and beacon counts
    grid_sizes = sorted(set(r['config']['grid_width'] for r in all_results))
    beacon_counts = sorted(set(r['config']['num_beacons'] for r in all_results))
    
    # Create heatmap data
    heatmap_data = np.full((len(beacon_counts), len(grid_sizes)), np.nan)
    
    for r in all_results:
        grid_idx = grid_sizes.index(r['config']['grid_width'])
        beacon_idx = beacon_counts.index(r['config']['num_beacons'])
        heatmap_data[beacon_idx, grid_idx] = r['mean_error']
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(grid_sizes)))
    ax.set_yticks(np.arange(len(beacon_counts)))
    ax.set_xticklabels(grid_sizes)
    ax.set_yticklabels(beacon_counts)
    
    ax.set_xlabel('Grid Size (meters)', fontsize=12)
    ax.set_ylabel('Number of Beacons', fontsize=12)
    ax.set_title('Localization Error vs Environment Configuration', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Localization Error (m)')
    
    # Add text annotations
    for i in range(len(beacon_counts)):
        for j in range(len(grid_sizes)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_error_vs_parameter(metrics: DomainGeneralizationMetrics, param_name: str):
    """Plot localization error vs a specific environment parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_metrics = metrics.get_metrics_by_environment_param(param_name)
    
    if not param_metrics:
        print(f"No data for parameter: {param_name}")
        return fig
    
    # Sort by key
    sorted_keys = sorted(param_metrics.keys())
    means = [param_metrics[k]['mean'] for k in sorted_keys]
    stds = [param_metrics[k]['std'] for k in sorted_keys]
    
    # Plot with error bars
    ax.errorbar(sorted_keys, means, yerr=stds, fmt='o-', capsize=5, 
               markersize=8, linewidth=2, label='Mean ± Std')
    
    # Labels based on parameter
    if param_name == 'grid_size':
        xlabel = 'Grid Size (meters)'
        title = 'Localization Error vs Grid Size'
    elif param_name == 'beacon_count':
        xlabel = 'Number of Beacons'
        title = 'Localization Error vs Beacon Count'
    elif param_name == 'los_prob':
        xlabel = 'LoS Probability'
        title = 'Localization Error vs LoS Probability'
    else:
        xlabel = param_name
        title = f'Localization Error vs {param_name}'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Localization Error (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(all_results: list):
    """Plot distribution of localization errors across all evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_errors = [r['mean_error'] for r in all_results]
    
    # Histogram
    axes[0].hist(all_errors, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(all_errors), color='r', linestyle='--', label=f'Mean: {np.mean(all_errors):.3f}m')
    axes[0].axvline(np.median(all_errors), color='g', linestyle='--', label=f'Median: {np.median(all_errors):.3f}m')
    axes[0].set_xlabel('Mean Localization Error (m)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Localization Errors', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(all_errors, vert=True)
    axes[1].set_ylabel('Localization Error (m)', fontsize=12)
    axes[1].set_title('Localization Error Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_reward_vs_error(all_results: list):
    """Plot relationship between reward and localization error."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = np.array([r['mean_reward'] for r in all_results])
    errors = np.array([r['mean_error'] for r in all_results])
    
    scatter = ax.scatter(errors, means, c=range(len(all_results)), cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black')
    
    # Add trend line
    z = np.polyfit(errors, means, 2)
    p = np.poly1d(z)
    sorted_errors = np.sort(errors)
    ax.plot(sorted_errors, p(sorted_errors), "r--", linewidth=2, label='Trend')
    
    ax.set_xlabel('Mean Localization Error (m)', fontsize=12)
    ax.set_ylabel('Mean Cumulative Reward', fontsize=12)
    ax.set_title('Reward vs Localization Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Configuration Index')
    
    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*70)
    print("DOMAIN GENERALIZATION EVALUATION")
    print("="*70 + "\n")
    
    # =========================================================================
    # ENABLE CIR-BASED DISTANCE MEASUREMENTS FOR EVALUATION
    # =========================================================================
    from rl.cir_training_config import setup_cir_training, FAST_TRAINING
    setup_cir_training(FAST_TRAINING)
    print()
    # =========================================================================
    
    # Load domain-generalized model
    print("Loading domain-generalized DQN model...")
    state_size = NUM_BEACONS
    domain_gen_trainer = DQNTrainer(state_size=state_size)
    
    # Find most recent model in checkpoints/domain_generalization
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints' / 'domain_generalization'
    
    if checkpoint_dir.exists():
        model_files = sorted(checkpoint_dir.glob('dqn_domain_generalization_*.pt'), reverse=True)
        if model_files:
            model_path = model_files[0]
            domain_gen_trainer.load_model(str(model_path))
            domain_gen_trainer.epsilon = 0.0
            print(f"Domain-Generalized Model loaded from {model_path}\n")
        else:
            print(f"Warning: No domain generalization models found in {checkpoint_dir}\n")
            return
    else:
        print(f"Warning: Checkpoint directory not found at {checkpoint_dir}\n")
        print("Run train_domain_generalization.py first to create the model\n")
        return
    
    # Load baseline single-environment model for comparison
    print("Loading baseline single-environment DQN model...")
    baseline_trainer = DQNTrainer(state_size=state_size)
    baseline_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
    
    if baseline_path.exists():
        baseline_trainer.load_model(str(baseline_path))
        baseline_trainer.epsilon = 0.0
        print(f"Baseline Model loaded from {baseline_path}\n")
    else:
        print("Warning: Baseline model not found. Skipping comparison.\n")
        baseline_trainer = None
    
    # Generate evaluation configurations
    print("Generating evaluation configurations...")
    eval_configs = generate_evaluation_configs(num_configs=50, seed=999)
    print(f"Generated {len(eval_configs)} diverse evaluation configurations\n")
    
    # Evaluate domain-generalized model
    print("="*70)
    print("Evaluating Domain-Generalized Model")
    print("="*70 + "\n")
    
    metrics = DomainGeneralizationMetrics()
    all_results = []
    
    for config in tqdm(eval_configs, desc="Evaluating configurations"):
        result = evaluate_on_config(domain_gen_trainer, config, num_episodes=5)
        all_results.append(result)
        
        metrics.add_episode_result(
            config['config_id'],
            config,
            result['mean_error'],
            result['mean_reward'],
            np.mean(result['episode_lengths'])
        )
    
    # Get summary metrics
    summary = metrics.get_summary_metrics()
    
    print("\n" + "="*70)
    print("DOMAIN GENERALIZATION EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    print("Overall Performance:")
    print(f"  Mean Error:       {summary['mean_error']:.4f} m")
    print(f"  Std Error:        {summary['std_error']:.4f} m")
    print(f"  Median Error:     {summary['median_error']:.4f} m")
    print(f"  90th Percentile:  {summary['error_90th']:.4f} m")
    print(f"  95th Percentile:  {summary['error_95th']:.4f} m")
    print(f"  Mean Reward:      {summary['mean_reward']:.4f}")
    print(f"  Avg Episode Len:  {summary['mean_trajectory_length']:.1f} steps\n")
    
    # Performance by environment parameter
    print("Performance by Grid Size:")
    grid_metrics = metrics.get_metrics_by_environment_param('grid_size')
    for size in sorted(grid_metrics.keys()):
        print(f"  {size}m: {grid_metrics[size]['mean']:.4f} m ± {grid_metrics[size]['std']:.4f} m (n={grid_metrics[size]['count']})")
    
    print("\nPerformance by Beacon Count:")
    beacon_metrics = metrics.get_metrics_by_environment_param('beacon_count')
    for count in sorted(beacon_metrics.keys()):
        print(f"  {count} beacons: {beacon_metrics[count]['mean']:.4f} m ± {beacon_metrics[count]['std']:.4f} m (n={beacon_metrics[count]['count']})")
    
    print("\nPerformance by LoS Probability:")
    los_metrics = metrics.get_metrics_by_environment_param('los_prob')
    for prob in sorted(los_metrics.keys()):
        print(f"  {prob:.1f} LoS prob: {los_metrics[prob]['mean']:.4f} m ± {los_metrics[prob]['std']:.4f} m (n={los_metrics[prob]['count']})")
    
    # Baseline comparison
    if baseline_trainer is not None:
        print("\n" + "="*70)
        print("BASELINE COMPARISON (Single-Environment Model)")
        print("="*70 + "\n")
        
        baseline_metrics = DomainGeneralizationMetrics()
        baseline_results = []
        
        for config in tqdm(eval_configs[:10], desc="Evaluating baseline on subset"):
            result = evaluate_on_config(baseline_trainer, config, num_episodes=3)
            baseline_results.append(result)
            baseline_metrics.add_episode_result(
                config['config_id'],
                config,
                result['mean_error'],
                result['mean_reward'],
                np.mean(result['episode_lengths'])
            )
        
        baseline_summary = baseline_metrics.get_summary_metrics()
        
        print("Baseline Performance (on 10 configurations):")
        print(f"  Mean Error:       {baseline_summary['mean_error']:.4f} m")
        print(f"  Std Error:        {baseline_summary['std_error']:.4f} m")
        print(f"  Mean Reward:      {baseline_summary['mean_reward']:.4f}\n")
        
        # Improvement
        improvement = (baseline_summary['mean_error'] - summary['mean_error']) / baseline_summary['mean_error'] * 100
        print(f"Domain Gen Improvement: {improvement:+.2f}%")
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating visualization plots...")
    print("="*70 + "\n")
    
    figs = {}
    plot_dir = Path(__file__).parent / 'results' / 'domain_generalization'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Error vs Environment Parameters
    print("  - Error vs Grid Size")
    figs['error_vs_grid'] = plot_error_vs_parameter(metrics, 'grid_size')
    
    print("  - Error vs Beacon Count")
    figs['error_vs_beacons'] = plot_error_vs_parameter(metrics, 'beacon_count')
    
    print("  - Error vs LoS Probability")
    figs['error_vs_los'] = plot_error_vs_parameter(metrics, 'los_prob')
    
    print("  - Error Distribution")
    figs['error_dist'] = plot_error_distribution(all_results)
    
    print("  - Reward vs Error")
    figs['reward_vs_error'] = plot_reward_vs_error(all_results)
    
    print("  - Heatmap")
    figs['heatmap'] = plot_generalization_heatmap(all_results)
    
    # Save plots
    plot_names = {
        'error_vs_grid': 'error_vs_grid_size.png',
        'error_vs_beacons': 'error_vs_beacon_count.png',
        'error_vs_los': 'error_vs_los_probability.png',
        'error_dist': 'error_distribution.png',
        'reward_vs_error': 'reward_vs_error.png',
        'heatmap': 'generalization_heatmap.png',
    }
    
    for key, fig in figs.items():
        save_path = plot_dir / plot_names[key]
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        plt.close(fig)
    
    # Save detailed results to JSON
    results_file = plot_dir / 'domain_generalization_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'grid_metrics': {str(k): v for k, v in grid_metrics.items()},
            'beacon_metrics': {str(k): v for k, v in beacon_metrics.items()},
            'los_metrics': {str(k): v for k, v in los_metrics.items()},
            'all_results': [{
                'config_id': r['config_id'],
                'config': r['config'],
                'mean_error': float(r['mean_error']),
                'std_error': float(r['std_error']),
                'mean_reward': float(r['mean_reward']),
            } for r in all_results]
        }, f, indent=2)
    print(f"\n  Saved: {results_file}")
    
    print("\n" + "="*70)
    print(f"Domain generalization evaluation complete!")
    print(f"Results saved to {plot_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
