import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from rl.trainer_dqn import DQNTrainer
from localization.trilateration import trilateration_2d, compute_noisy_distances, localization_error
from config import NUM_BEACONS, NUM_SELECTED_BEACONS


class EvaluationMetrics:
    """Collect and compute evaluation metrics."""
    
    def __init__(self):
        self.localization_errors = []
        self.rewards = []
        self.battery_levels = defaultdict(list)
        self.selected_beacons_history = []
        self.los_selections = 0
        self.nlos_selections = 0
        self.time_steps = 0
        self.beacon_counts = np.zeros(NUM_BEACONS)
    
    def add_error(self, error):
        self.localization_errors.append(error)
    
    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def add_battery_levels(self, levels):
        for i, level in enumerate(levels):
            self.battery_levels[i].append(level)
    
    def add_selected_beacons(self, indices, los_flags):
        self.selected_beacons_history.append(indices)
        for idx in indices:
            self.beacon_counts[idx] += 1
            if los_flags[idx]:
                self.los_selections += 1
            else:
                self.nlos_selections += 1
    
    def get_metrics(self):
        """Return computed metrics."""
        metrics = {
            'mean_error': np.mean(self.localization_errors),
            'rmse_error': np.sqrt(np.mean(np.array(self.localization_errors) ** 2)),
            'std_error': np.std(self.localization_errors),
            'min_error': np.min(self.localization_errors),
            'max_error': np.max(self.localization_errors),
            'error_90th': np.percentile(self.localization_errors, 90),
            'error_95th': np.percentile(self.localization_errors, 95),
            'mean_reward': np.mean(self.rewards),
            'total_steps': self.time_steps,
            'final_batteries': {i: self.battery_levels[i][-1] for i in range(NUM_BEACONS)},
            'battery_deviation': self._compute_battery_deviation(),
            'los_ratio': self.los_selections / (self.los_selections + self.nlos_selections) if (self.los_selections + self.nlos_selections) > 0 else 0,
            'selection_frequency': self.beacon_counts / (np.sum(self.beacon_counts) if np.sum(self.beacon_counts) > 0 else 1),
            'localization_errors': self.localization_errors,
            'battery_levels': dict(self.battery_levels),
            'rewards': self.rewards,
        }
        return metrics
    
    def _compute_battery_deviation(self):
        """Compute mean squared deviation of battery levels."""
        final_batteries = np.array([self.battery_levels[i][-1] for i in range(NUM_BEACONS)])
        mean_battery = np.mean(final_batteries)
        return np.mean(((final_batteries - mean_battery) / (mean_battery + 1e-6)) ** 2)


def random_selection(env: Environment) -> list:
    """Random beacon selection."""
    return list(np.random.choice(NUM_BEACONS, NUM_SELECTED_BEACONS, replace=False))


def nearest_neighbor_selection(env: Environment) -> list:
    """Nearest neighbor beacon selection."""
    agent_pos = np.array(env.agent.get_position())
    beacon_positions = np.array([beacon.position for beacon in env.beacons])
    
    # Compute distances
    distances = np.linalg.norm(beacon_positions - agent_pos, axis=1)
    
    # Select nearest 3 beacons
    return list(np.argsort(distances)[:NUM_SELECTED_BEACONS])


def rl_selection(env: Environment, trainer: DQNTrainer) -> list:
    """RL-based beacon selection (DQN)."""
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


def evaluate_method(method_name: str, 
                   selection_func,
                   trainer: DQNTrainer = None,
                   num_epochs: int = 100,
                   seed_offset: int = 0) -> EvaluationMetrics:
    """
    Evaluate a beacon selection method.
    
    Args:
        method_name: Name of the method
        selection_func: Function to select beacons
        trainer: DQN trainer (for DQN method)
        num_epochs: Number of epochs
        seed_offset: Offset for random seed to ensure different epochs have different randomness
    
    Returns:
        EvaluationMetrics object
    """
    metrics = EvaluationMetrics()
    
    pbar = tqdm(range(num_epochs), desc=f"Evaluating {method_name}", position=0)
    
    for epoch in pbar:
        # Set seed based on epoch to ensure all methods see same environment sequence
        np.random.seed(epoch + seed_offset)
        torch.manual_seed(epoch + seed_offset)
        
        env = Environment()
        
        # Critical battery threshold: terminate when any beacon drops below this level
        CRITICAL_BATTERY_THRESHOLD = 10.0  # percent
        
        for step in range(100):
            # Select beacons FIRST based on current state (before environment transitions)
            if trainer is not None:
                selected_indices = selection_func(env, trainer)
            else:
                selected_indices = selection_func(env)
            
            # Apply beacon selection to environment
            env.selected_beacon_indices = selected_indices
            
            # NOW step the environment with the applied action
            env.step()
            
            # Compute metrics on the new state AFTER environment transition
            agent_pos = np.array(env.agent.get_position())
            selected_positions = np.array([env.beacons[i].position for i in selected_indices])
            los_flags = [env.current_links[i] for i in selected_indices]
            
            # Get noisy distances using same model as training
            distances = compute_noisy_distances(agent_pos, selected_positions, los_flags)
            
            # Estimate position via trilateration
            est_x, est_y = trilateration_2d(selected_positions, distances)
            est_pos = np.array([est_x, est_y])
            
            # Compute error: ||ground_truth - estimated||
            error = np.sqrt(np.sum((agent_pos - est_pos) ** 2))
            metrics.add_error(error)
            
            # Compute reward
            reward = compute_reward(agent_pos, selected_positions, los_flags, env.get_battery_levels())
            metrics.add_reward(reward)
            
            # Track metrics
            metrics.add_battery_levels(env.get_battery_levels())
            metrics.add_selected_beacons(selected_indices, env.current_links)
            
            # Early termination: if any beacon's battery drops below critical threshold
            battery_levels = env.get_battery_levels()
            min_battery = min(battery_levels)
            if min_battery <= CRITICAL_BATTERY_THRESHOLD:
                # Episode ends when system degrades (any beacon reaches critical level)
                break
            
        metrics.time_steps += 1
        
        pbar.update(1)
    
    pbar.close()
    return metrics


def plot_ecdf_comparison(results: dict):
    """Plot ECDF of localization errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, metrics in results.items():
        errors = sorted(metrics['localization_errors'])
        ecdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, ecdf, marker='o', linestyle='-', label=method_name, markersize=3, alpha=0.7)
    
    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('ECDF of Localization Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_error_comparison(results: dict):
    """Plot mean and RMSE error comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(results.keys())
    mean_errors = [results[m]['mean_error'] for m in methods]
    rmse_errors = [results[m]['rmse_error'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, mean_errors, width, label='Mean Error', alpha=0.8)
    ax1.bar(x + width/2, rmse_errors, width, label='RMSE Error', alpha=0.8)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Error (m)', fontsize=12)
    ax1.set_title('Localization Error Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    errors_list = [results[m]['localization_errors'] for m in methods]
    ax2.boxplot(errors_list, labels=methods)
    ax2.set_ylabel('Localization Error (m)', fontsize=12)
    ax2.set_title('Localization Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_battery_levels(results: dict):
    """Plot remaining battery levels over time."""
    # Create dynamic grid based on number of beacons
    num_cols = 3
    num_rows = (NUM_BEACONS + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    
    methods = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for beacon_id in range(NUM_BEACONS):
        ax = axes[beacon_id]
        
        for method_name, color in zip(methods, colors):
            battery_history = results[method_name]['battery_levels'].get(beacon_id, [])
            ax.plot(battery_history, label=method_name, color=color, linewidth=2)
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Battery Level (%)', fontsize=10)
        ax.set_title(f'Beacon {beacon_id} Battery Level', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    return fig


def plot_battery_deviation(results: dict):
    """Plot battery deviation comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    deviations = [results[m]['battery_deviation'] for m in methods]
    
    bars = ax.bar(methods, deviations, alpha=0.8, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    
    ax.set_ylabel('Battery Deviation (MSE)', fontsize=12)
    ax.set_title('Battery Level Deviation Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_infrastructure_lifetime(results: dict):
    """Plot infrastructure lifetime (time to first beacon depletion)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    lifetimes = []
    
    for method_name in methods:
        battery_data = results[method_name]['battery_levels']
        # Find first depletion across all beacons
        min_lifetime = float('inf')
        for beacon_id in range(NUM_BEACONS):
            history = battery_data.get(beacon_id, [])
            if history:
                # Find when battery first drops below 10%
                depletion_idx = next((i for i, b in enumerate(history) if b < 10), len(history))
                min_lifetime = min(min_lifetime, depletion_idx)
        
        lifetimes.append(min_lifetime if min_lifetime != float('inf') else len(history))
    
    bars = ax.bar(methods, lifetimes, alpha=0.8, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    
    ax.set_ylabel('Time Steps', fontsize=12)
    ax.set_title('Infrastructure Lifetime (Time to First Beacon Depletion)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_cumulative_reward(results: dict):
    """Plot cumulative reward vs epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, metrics in results.items():
        cumulative_rewards = np.cumsum(metrics['rewards'])
        ax.plot(cumulative_rewards, label=method_name, linewidth=2)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Cumulative Reward vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_los_ratio_comparison(results: dict):
    """Plot LoS vs NLoS selection ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    los_ratios = [results[m]['los_ratio'] for m in methods]
    nlos_ratios = [1 - r for r in los_ratios]
    
    x = np.arange(len(methods))
    width = 0.6
    
    ax.bar(x, los_ratios, width, label='LoS', alpha=0.8)
    ax.bar(x, nlos_ratios, width, bottom=los_ratios, label='NLoS', alpha=0.8)
    
    ax.set_ylabel('Selection Ratio', fontsize=12)
    ax.set_title('LoS vs NLoS Selection Ratio Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    
    # Add percentage labels
    for i, (los, nlos) in enumerate(zip(los_ratios, nlos_ratios)):
        ax.text(i, los/2, f'{los*100:.1f}%', ha='center', va='center', fontweight='bold')
        ax.text(i, los + nlos/2, f'{nlos*100:.1f}%', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*70)
    print("BEACON SELECTION EVALUATION - 3 METHODS")
    print("="*70 + "\n")
    
    # Load trained DQN model
    print("Loading trained DQN model...")
    state_size = NUM_BEACONS  # Only battery levels (observable), no ground-truth position or LoS flags
    trainer = DQNTrainer(state_size=state_size)
    dqn_model_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
    
    if dqn_model_path.exists():
        trainer.load_model(str(dqn_model_path))
        trainer.epsilon = 0.0  # No exploration during evaluation
        print(f"DQN Model loaded from {dqn_model_path}")
    else:
        print(f"Warning: Model not found at {dqn_model_path}")
        print("Using untrained model for evaluation")
    
    # Evaluate methods on same simulations (same seeds)
    results = {}
    
    print("Evaluating all methods on same simulation sequences...\n")
    seed_offset = 42  # Base seed for reproducibility
    
    print("Evaluating Random Node Selection...")
    results['Random'] = evaluate_method('Random', random_selection, num_epochs=1000, seed_offset=seed_offset)
    
    print("\nEvaluating Nearest Neighbor...")
    results['Nearest Neighbor'] = evaluate_method('Nearest Neighbor', nearest_neighbor_selection, num_epochs=1000, seed_offset=seed_offset)
    
    print("\nEvaluating DQN-based Selection...")
    results['DQN'] = evaluate_method('DQN', rl_selection, trainer=trainer, num_epochs=1000, seed_offset=seed_offset)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    # Convert EvaluationMetrics objects to dictionaries
    results_dict = {}
    for method_name, metrics in results.items():
        results_dict[method_name] = metrics.get_metrics()
    
    for method_name, metric_dict in results_dict.items():
        print(f"{method_name}:")
        print(f"  Mean Localization Error: {metric_dict['mean_error']:.4f} m")
        print(f"  RMSE Localization Error: {metric_dict['rmse_error']:.4f} m")
        print(f"  90th Percentile Error:   {metric_dict['error_90th']:.4f} m")
        print(f"  95th Percentile Error:   {metric_dict['error_95th']:.4f} m")
        print(f"  Std Dev Error: {metric_dict['std_error']:.4f} m")
        print(f"  Mean Reward: {metric_dict['mean_reward']:.4f}")
        print(f"  Total Steps: {metric_dict['total_steps']}")
        print(f"  Battery Deviation: {metric_dict['battery_deviation']:.6f}")
        print(f"  LoS Selection Ratio: {metric_dict['los_ratio']:.2%}")
        
        # Print selection frequency top beacons
        freq = metric_dict['selection_frequency']
        top_indices = np.argsort(freq)[::-1][:5]
        print(f"  Top 5 Selected Beacons: {', '.join([f'B{i}({freq[i]:.1%})' for i in top_indices])}")
        print()
    
    # Save results to text file
    eval_dir = Path(__file__).parent / 'results'
    eval_dir.mkdir(exist_ok=True)
    summary_file = eval_dir / 'evaluation_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("BEACON SELECTION EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for method_name, metric_dict in results_dict.items():
            f.write(f"{method_name}:\n")
            f.write(f"  Mean Localization Error: {metric_dict['mean_error']:.4f} m\n")
            f.write(f"  RMSE Localization Error: {metric_dict['rmse_error']:.4f} m\n")
            f.write(f"  90th Percentile Error:   {metric_dict['error_90th']:.4f} m\n")
            f.write(f"  95th Percentile Error:   {metric_dict['error_95th']:.4f} m\n")
            f.write(f"  Std Dev Error:           {metric_dict['std_error']:.4f} m\n")
            f.write(f"  Mean Reward:             {metric_dict['mean_reward']:.4f}\n")
            f.write(f"  Total Steps:             {metric_dict['total_steps']}\n")
            f.write(f"  Battery Deviation:       {metric_dict['battery_deviation']:.6f}\n")
            f.write(f"  LoS Selection Ratio:     {metric_dict['los_ratio']:.2%}\n")
            
            f.write("  Selection Frequency per Beacon:\n")
            freq = metric_dict['selection_frequency']
            for i, p in enumerate(freq):
                f.write(f"    Beacon {i}: {p:.1%}\n")
            f.write("\n")
            
    print(f"Summary saved to {summary_file}")
    
    # Generate plots
    print("Generating plots...")
    
    figs = {
        'ecdf': plot_ecdf_comparison(results_dict),
        'error': plot_error_comparison(results_dict),
        'battery': plot_battery_levels(results_dict),
        'deviation': plot_battery_deviation(results_dict),
        'lifetime': plot_infrastructure_lifetime(results_dict),
        'reward': plot_cumulative_reward(results_dict),
        'los_ratio': plot_los_ratio_comparison(results_dict),
    }
    
    # Save plots
    eval_dir = Path(__file__).parent / 'results'
    eval_dir.mkdir(exist_ok=True)
    
    plot_names = {
        'ecdf': 'ecdf_localization_error.png',
        'error': 'error_comparison.png',
        'battery': 'battery_levels.png',
        'deviation': 'battery_deviation.png',
        'lifetime': 'infrastructure_lifetime.png',
        'reward': 'cumulative_reward.png',
        'los_ratio': 'los_ratio_comparison.png',
    }
    
    for key, fig in figs.items():
        save_path = eval_dir / plot_names[key]
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    print(f"\n" + "="*70)
    print(f"Evaluation complete! Results saved to {eval_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()