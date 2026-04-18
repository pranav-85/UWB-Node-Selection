import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from rl.trainer_dqn import DQNTrainer
from rl.trainer_lstm import LSTMTrainer
from rl.trainer_enhanced_lstm import EnhancedDQNTrainer
from rl.trainer_ppo import PPOTrainer
from localization.gdop import compute_weighted_gdop
from localization.trilateration import trilateration_2d, compute_noisy_distances, localization_error
from localization.wls_kalman import WLSLocalizer
from config import NUM_BEACONS, NUM_SELECTED_BEACONS, BEACON_POSITIONS, GRID_SIZE
from rl.train_meta_rl import MetaDQN
from rl.train_rl2_lstm import LSTM_DQN, STATE_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, POSSIBLE_ACTIONS, get_state


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
        self.gdop_values = []
        self.trajectories = []  # List of (x, y) tuples for each step in each episode
    
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

    def add_trajectory(self, trajectory):
        """Add a trajectory (list of positions) for an episode."""
        self.trajectories.append(trajectory)
    
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
            'trajectories': self.trajectories,
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

def lstm_selection(env: Environment, trainer: LSTMTrainer) -> list:
    """RL-based beacon selection (LSTM)."""
    state_seq = trainer.replay_buffer.get_state_sequence()
    action = trainer.select_action(state_seq, training=False)
    return list(trainer.possible_actions[action])

def ppo_selection(env: Environment, trainer: PPOTrainer) -> list:
    """RL-based beacon selection (PPO)."""
    state = trainer.state_to_vector(env)
    action, _, _ = trainer.select_action(state)
    return list(trainer.possible_actions[action])

def enhanced_lstm_selection(env: Environment, trainer: EnhancedDQNTrainer) -> list:
    """RL-based beacon selection with Enhanced DQN (WLS + geometry features)."""
    # Build enhanced state based on current environment
    # Use last known selected beacons or default to first 3
    selected = (0, 1, 2)
    state = trainer.build_enhanced_state(env, list(selected))
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])

def wgdop_selection(env: Environment) -> list:
    """
    Weighted GDOP-based beacon selection.
    Real-time style (uses only current observable state).
    """

    agent_pos = np.array(env.agent.get_position())
    best_score = float('inf')
    best_combo = None

    # Try all combinations of 3 beacons
    for combo in combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS):

        selected_positions = [
            env.beacons[i].position for i in combo
        ]

        selected_los_flags = [
            env.current_links[i] for i in combo
        ]

        score = compute_weighted_gdop(
            agent_estimate=agent_pos,
            beacon_positions=selected_positions,
            los_flags=selected_los_flags
        )

        if score < best_score:
            best_score = score
            best_combo = combo

    return list(best_combo)

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
    print(f"    [EVAL] Starting evaluation of {method_name}")
    
    pbar = tqdm(range(num_epochs), desc=f"Evaluating {method_name}", position=0)
    
    for epoch in pbar:
        # Set seed based on epoch to ensure all methods see same environment sequence
        np.random.seed(epoch + seed_offset)
        torch.manual_seed(epoch + seed_offset)
        
        env = Environment()
        
        # Initialize trainer state/history for this epoch
        if trainer is not None:
            if isinstance(trainer, RL2LSTMWrapper):
                # For RL² LSTM: reset hidden state at start of each episode
                trainer.reset_hidden_state()
            elif hasattr(trainer, 'build_enhanced_state'):
                # For Enhanced LSTM trainer
                initial_state = trainer.build_enhanced_state(env, [0, 1, 2])
                trainer.replay_buffer.state_history.clear()
                for _ in range(trainer.seq_length):
                    trainer.replay_buffer.update_state_history(initial_state)
                # Initialize localizer
                if hasattr(trainer, 'localizer'):
                    trainer.localizer.reset()
            elif hasattr(trainer, 'seq_length') and hasattr(trainer.replay_buffer, 'state_history'):
                # For regular LSTM trainer (check for seq_length attribute and state_history)
                initial_state = trainer.state_to_vector(env)
                trainer.replay_buffer.state_history.clear()
                for _ in range(trainer.seq_length):
                    trainer.replay_buffer.update_state_history(initial_state)
        
        # Critical battery threshold: terminate when any beacon drops below this level
        CRITICAL_BATTERY_THRESHOLD = 10.0  # percent
        
        current_trajectory = []
        estimated_pos = np.mean([env.beacons[i].position for i in range(3)], axis=0)

        for step in range(num_epochs):
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
            current_trajectory.append(tuple(agent_pos))

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
            
            # Update state history for next iteration (only for LSTM trainers)
            if trainer is not None:
                if hasattr(trainer, 'build_enhanced_state'):
                    # Enhanced LSTM trainer: estimate position and build enhanced state
                    next_state = trainer.build_enhanced_state(env, selected_indices, estimated_pos)
                    trainer.replay_buffer.update_state_history(next_state)
                    # For next iteration, update estimated position reference
                    estimated_pos = agent_pos
                elif hasattr(trainer, 'seq_length') and hasattr(trainer.replay_buffer, 'state_history'):
                    # Regular LSTM trainer
                    next_state = trainer.state_to_vector(env)
                    trainer.replay_buffer.update_state_history(next_state)
            
            # Early termination: if any beacon's battery drops below critical threshold
            battery_levels = env.get_battery_levels()
            min_battery = min(battery_levels)
            if min_battery <= CRITICAL_BATTERY_THRESHOLD:
                # Episode ends when system degrades (any beacon reaches critical level)
                break
            
        metrics.time_steps += 1
        metrics.add_trajectory(current_trajectory)
        
        pbar.update(1)
    
    pbar.close()
    print(f"    [EVAL] [OK] {method_name} evaluation complete - collected {len(metrics.localization_errors)} errors")
    return metrics


def plot_ecdf_comparison(results: dict):
    """Plot improved ECDF of localization errors."""
    print(f"\n[PLOT-ECDF] Starting ECDF plot with {len(results)} methods")
    print(f"[PLOT-ECDF] Methods: {list(results.keys())}")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors  # clean professional color set

    for idx, (method_name, metrics) in enumerate(results.items()):
        errors = np.sort(metrics['localization_errors'])
        ecdf = np.arange(1, len(errors) + 1) / len(errors)
        
        print(f"[PLOT-ECDF] {idx+1}. Plotting {method_name}: {len(errors)} errors, color_idx={idx % len(colors)}")

        ax.plot(
            errors,
            ecdf,
            linewidth=2.5,
            label=method_name,
            color=colors[idx % len(colors)]
        )

    # Add reference percentile lines
    ax.axhline(0.9, linestyle='--', color='gray', alpha=0.6)
    ax.axhline(0.95, linestyle='--', color='gray', alpha=0.6)

    ax.set_xlabel('Localization Error (m)', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('ECDF of Localization Error Comparison', fontsize=15, fontweight='bold')

    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    
    print(f"[PLOT-ECDF] [OK] ECDF plot complete with {len(results)} lines")
    print(f"[PLOT-ECDF] Legend includes: {[line.get_label() for line in ax.get_lines()][:5]}...")
    
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


    return fig


def plot_agent_movement(results: dict):
    """Plot agent movement trajectories."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot beacons
    for i, pos in enumerate(BEACON_POSITIONS):
        ax.plot(pos[0], pos[1], 'bs', markersize=10, label='Beacon' if i == 0 else "")
        ax.text(pos[0], pos[1] + 0.3, f'B{i}', ha='center', fontsize=12, fontweight='bold')
        
    # Plot trajectories from the 'Random' method (since agent movement is consistent across methods)
    # We'll plot a few trajectories to show coverage
    if 'Random' in results:
        trajectories = results['Random']['trajectories']
        # Plot last 5 trajectories
        for traj in trajectories[-10:]:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3)
            # Mark start and end
            if len(traj) > 0:
                ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6, label='Start' if traj is trajectories[-1] else "")
                ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=6, label='End' if traj is trajectories[-1] else "")
            
    ax.set_title(f'Agent Movement Trajectories (Last 10 Episodes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.grid(True, alpha=0.3)
    
    # Legend - handle duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    return fig


def domain_gen_selection(env: Environment, trainer: DQNTrainer) -> list:
    """RL-based beacon selection using domain-generalized DQN model."""
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


def meta_rl_selection(env: Environment, meta_model: nn.Module) -> list:
    """
    RL-based beacon selection using meta-learned DQN model.
    Uses the trained meta-model for quick decision-making on any environment.
    
    Args:
        env: Environment instance
        meta_model: Trained MetaDQN model
    
    Returns:
        List of selected beacon indices
    """
    # Get device from model parameters
    device = next(meta_model.parameters()).device
    
    # Get state vector (battery levels only)
    state = torch.tensor(np.array(env.get_battery_levels(), dtype=np.float32)).unsqueeze(0).to(device)
    
    # Get Q-values from meta-model
    with torch.no_grad():
        q_values = meta_model(state)
        action_idx = q_values.argmax(dim=1).item()
    
    # Convert action index to beacon combination
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    return list(possible_actions[action_idx])


class RL2LSTMWrapper:
    """Wrapper for RL² LSTM model that manages hidden state during evaluation."""
    
    def __init__(self, model: LSTM_DQN, device: torch.device):
        """
        Args:
            model: Trained LSTM_DQN model
            device: Device to use (cpu or cuda)
        """
        self.model = model
        self.device = device
        self.hidden_state = None
        self.possible_actions = POSSIBLE_ACTIONS
    
    def reset_hidden_state(self):
        """Reset hidden state for new episode."""
        self.hidden_state = None
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using LSTM-DQN with persistent hidden state.
        
        Args:
            state: Current state vector (battery levels)
        
        Returns:
            Action index (beacon combination)
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values, self.hidden_state = self.model(state_tensor, self.hidden_state)
            action = q_values.argmax(dim=-1).item()
        
        return action


def rl2_lstm_selection(env: Environment, wrapper: RL2LSTMWrapper) -> list:
    """
    RL² LSTM-based beacon selection.
    Uses LSTM hidden state for implicit environment adaptation.
    
    Args:
        env: Environment instance
        wrapper: RL2LSTMWrapper with model and hidden state
    
    Returns:
        List of selected beacon indices
    """
    state = get_state(env)
    action_idx = wrapper.select_action(state)
    return list(wrapper.possible_actions[action_idx])


def main():
    print("\n" + "="*70)
    print("BEACON SELECTION EVALUATION")
    print("="*70 + "\n")
    
    # =========================================================================
    # ENABLE CIR-BASED DISTANCE MEASUREMENTS FOR EVALUATION
    # =========================================================================
    from rl.cir_training_config import setup_cir_training, FAST_TRAINING
    setup_cir_training(FAST_TRAINING)
    print()
    # =========================================================================
    
    # Load trained DQN model
    print("Loading trained DQN model...")
    state_size = NUM_BEACONS  # Only battery levels (observable), no ground-truth position or LoS flags
    dqn_trainer = DQNTrainer(state_size=state_size)
    dqn_model_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
    
    if dqn_model_path.exists():
        dqn_trainer.load_model(str(dqn_model_path))
        dqn_trainer.epsilon = 0.0  # No exploration during evaluation
        print(f"DQN Model loaded from {dqn_model_path}")
    else:
        print(f"Warning: DQN model not found at {dqn_model_path}")
        print("Using untrained DQN model for evaluation")
    
    # Load trained LSTM model
    print("Loading trained LSTM model...")
    lstm_trainer = LSTMTrainer(
        state_size=state_size,
        lstm_hidden_size=64,
        fc_hidden_size=64,
        seq_length=10
    )
    lstm_model_path = Path(__file__).parent.parent / 'checkpoints' / 'lstm_model.pt'
    
    if lstm_model_path.exists():
        lstm_trainer.load_model(str(lstm_model_path))
        lstm_trainer.epsilon = 0.0  # No exploration during evaluation
        print(f"LSTM Model loaded from {lstm_model_path}")
    else:
        print(f"Warning: LSTM model not found at {lstm_model_path}")
        print("Using untrained LSTM model for evaluation")
    
    # Load trained Enhanced DQN model
    print("Loading trained Enhanced DQN model...")
    enhanced_dqn_trainer = EnhancedDQNTrainer(
        hidden_size=128
    )
    enhanced_model_path = Path(__file__).parent.parent / 'checkpoints' / 'enhanced_dqn_model.pt'
    
    if enhanced_model_path.exists():
        enhanced_dqn_trainer.load_model(str(enhanced_model_path))
        enhanced_dqn_trainer.epsilon = 0.0  # No exploration during evaluation
        print(f"Enhanced DQN Model loaded from {enhanced_model_path}")
    else:
        print(f"Warning: Enhanced DQN model not found at {enhanced_model_path}")
        print("Using untrained Enhanced DQN model for evaluation")
    
    # Load trained PPO model
    print("Loading trained PPO model...")
    ppo_trainer = PPOTrainer(
        state_size=NUM_BEACONS
    )
    ppo_model_path = Path(__file__).parent.parent / 'checkpoints' / 'ppo_model.pt'
    
    if ppo_model_path.exists():
        ppo_trainer.load_model(str(ppo_model_path))
        print(f"PPO Model loaded from {ppo_model_path}")
    else:
        print(f"Warning: PPO model not found at {ppo_model_path}")
        print("Using untrained PPO model for evaluation")
    
    # Evaluate methods on same simulations (same seeds)
    results = {}
    
    print("Evaluating all methods on same simulation sequences...\n")
    seed_offset = 42  # Base seed for reproducibility
    
    evaluated_methods = []  # Track which methods were successfully evaluated
    
    print("Evaluating Random Node Selection...")
    results['Random'] = evaluate_method('Random', random_selection, num_epochs=100, seed_offset=seed_offset)
    evaluated_methods.append('Random')
    
    print("Evaluating GDOP Node Selection...")
    results['GDOP'] = evaluate_method('GDOP', wgdop_selection, num_epochs=100, seed_offset=seed_offset)
    evaluated_methods.append('GDOP')

    print("\nEvaluating Nearest Neighbor...")
    results['Nearest Neighbor'] = evaluate_method('Nearest Neighbor', nearest_neighbor_selection, num_epochs=100, seed_offset=seed_offset)
    evaluated_methods.append('Nearest Neighbor')
    
    print("\nEvaluating DQN-based Selection...")
    results['DQN'] = evaluate_method('DQN', rl_selection, trainer=dqn_trainer, num_epochs=100, seed_offset=seed_offset)
    evaluated_methods.append('DQN')
    
    # Load and evaluate domain-generalized model if available
    print("\nChecking for domain-generalized model...")
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints' / 'domain_generalization'
    
    if checkpoint_dir.exists():
        model_files = sorted(checkpoint_dir.glob('dqn_domain_generalization_*.pt'), reverse=True)
        if model_files:
            model_path = model_files[0]
            print(f"Loading domain-generalized DQN model from {model_path}...")
            domain_gen_trainer = DQNTrainer(state_size=state_size)
            domain_gen_trainer.load_model(str(model_path))
            domain_gen_trainer.epsilon = 0.0
            
            print("Evaluating Domain-Generalized DQN Selection (benchmark evaluation)...")
            results['Domain Gen DQN'] = evaluate_method('Domain Gen DQN', domain_gen_selection, 
                                                       trainer=domain_gen_trainer, num_epochs=100, 
                                                       seed_offset=seed_offset)
            evaluated_methods.append('Domain Gen DQN')
        else:
            print("No domain generalization models found in checkpoint directory")
    else:
        print("Domain-generalized model checkpoint directory not found")
    
    # Load and evaluate meta-RL model if available
    print("\nChecking for meta-RL model...")
    meta_rl_checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints' / 'meta_rl'
    print(f"  Looking in: {meta_rl_checkpoint_dir}")
    
    if meta_rl_checkpoint_dir.exists():
        print(f"  Directory exists")
        all_files = list(meta_rl_checkpoint_dir.glob('*.pt'))
        print(f"  All .pt files in directory: {[f.name for f in all_files]}")
        
        meta_model_files = sorted(meta_rl_checkpoint_dir.glob('meta_dqn_final_*.pt'), reverse=True)
        if meta_model_files:
            model_path = meta_model_files[0]
            print(f"  Found meta-RL final model: {model_path.name}")
            print(f"Loading meta-RL model from {model_path}...")
            
            try:
                # Create MetaDQN model
                action_size = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))
                meta_model = MetaDQN(state_size=NUM_BEACONS, action_size=action_size, hidden_size=64)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                meta_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                meta_model.to(device)
                meta_model.eval()
                
                print(f"  [OK] Meta-RL model loaded successfully on {device}")
                print("Evaluating Meta-RL Selection...")
                results['Meta RL'] = evaluate_method('Meta RL', meta_rl_selection, 
                                                    trainer=meta_model, num_epochs=100, 
                                                    seed_offset=seed_offset)
                evaluated_methods.append('Meta RL')
            except Exception as e:
                print(f"  [FAIL] Error loading/evaluating Meta-RL model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [FAIL] No 'meta_dqn_final_*.pt' files found")
            print("  -> You need to run: python ./src/rl/train_meta_rl.py")
            checkpoint_files = list(meta_rl_checkpoint_dir.glob('meta_dqn_*.pt'))
            if checkpoint_files:
                print(f"  (Found {len(checkpoint_files)} checkpoint files but no final model)")
    else:
        print(f"  [FAIL] Directory does not exist: {meta_rl_checkpoint_dir}")
        print("  -> You need to run: python ./src/rl/train_meta_rl.py")
    
    # Load and evaluate RL² LSTM model if available
    print("\nChecking for RL² LSTM model...")
    rl2_lstm_checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints' / 'rl2_lstm'
    print(f"  Looking in: {rl2_lstm_checkpoint_dir}")
    
    if rl2_lstm_checkpoint_dir.exists():
        print(f"  [OK] Directory exists")
        all_files = list(rl2_lstm_checkpoint_dir.glob('*.pt'))
        print(f"  Found {len(all_files)} .pt files: {[f.name for f in sorted(all_files)[:10]]}")
        
        # Try multiple glob patterns to find model files
        rl2_model_files = list(rl2_lstm_checkpoint_dir.glob('rl2_lstm_*.pt'))
        print(f"  Glob 'rl2_lstm_*.pt' found: {len(rl2_model_files)} files")
        
        if not rl2_model_files:
            # Try without underscore variant
            rl2_model_files = list(rl2_lstm_checkpoint_dir.glob('rl2*lstm*.pt'))
            print(f"  Fallback glob 'rl2*lstm*.pt' found: {len(rl2_model_files)} files")
        
        if rl2_model_files:
            # Sort by episode number (highest first)
            rl2_model_files = sorted(rl2_model_files, 
                                    key=lambda x: int(x.name.split('_')[2]) if '_' in x.name else 0,
                                    reverse=True)
            
            model_path = rl2_model_files[0]
            print(f"  [OK] Found RL² LSTM model: {model_path.name}")
            print(f"  Loading from: {model_path}...")
            
            try:
                # Create and load RL² LSTM model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                rl2_lstm_model = LSTM_DQN(
                    input_size=STATE_SIZE,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    num_actions=len(POSSIBLE_ACTIONS),
                    num_layers=LSTM_NUM_LAYERS
                ).to(device)
                
                print(f"  [OK] Model created on {device}")
                print(f"    Expected: input={STATE_SIZE}, hidden={LSTM_HIDDEN_SIZE}, actions={len(POSSIBLE_ACTIONS)}")
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                print(f"  [OK] Checkpoint loaded from {model_path.name}")
                
                rl2_lstm_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  [OK] Model state dict loaded")
                
                rl2_lstm_model.eval()
                print(f"  [OK] RL² LSTM model loaded successfully on {device}")
                
                # Create wrapper for hidden state management
                rl2_wrapper = RL2LSTMWrapper(rl2_lstm_model, device)
                
                print("  Evaluating RL² LSTM Selection...")
                results['RL² LSTM'] = evaluate_method('RL² LSTM', rl2_lstm_selection, 
                                                     trainer=rl2_wrapper, num_epochs=100, 
                                                     seed_offset=seed_offset)
                evaluated_methods.append('RL² LSTM')
                print("\n  [OK] RL² LSTM evaluation complete!")
            except RuntimeError as e:
                if 'size mismatch' in str(e).lower():
                    print(f"  [FAIL] Model architecture mismatch - checkpoint incompatible")
                    print(f"     Old checkpoint uses different action space (4 movement actions)")
                    print(f"     New model needs: input={STATE_SIZE}, actions={len(POSSIBLE_ACTIONS)} (beacon combos)")
                    print(f"  -> Train new RL² LSTM model: python ./src/rl/train_rl2_lstm.py")
                else:
                    print(f"  [FAIL] RuntimeError loading RL² LSTM model: {e}")
            except Exception as e:
                print(f"  [FAIL] Error loading/evaluating RL² LSTM model: {e}")
                print(f"  Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        else:
            print("  [FAIL] No 'rl2_lstm_*.pt' files found")
            print("  -> Files should be at: checkpoints/rl2_lstm/rl2_lstm_N_TIMESTAMP.pt")
            print("  -> Run this first: python ./src/rl/train_rl2_lstm.py")
    else:
        print(f"  [FAIL] Directory does not exist: {rl2_lstm_checkpoint_dir}")
        print("  -> You need to run: python ./src/rl/train_rl2_lstm.py")
    
    # print("\nEvaluating LSTM-based Selection...")
    # results['LSTM'] = evaluate_method('LSTM', lstm_selection, trainer=lstm_trainer, num_epochs=100, seed_offset=seed_offset)
    
    # print("\nEvaluating PPO-based Selection...")
    # results['PPO'] = evaluate_method('PPO', ppo_selection, trainer=ppo_trainer, num_epochs=100, seed_offset=seed_offset)
    
    # print("\nEvaluating Enhanced DQN-based Selection (WLS + Geometry)...")
    # results['Enhanced DQN'] = evaluate_method('Enhanced DQN', enhanced_lstm_selection, trainer=enhanced_dqn_trainer, num_epochs=100, seed_offset=seed_offset)
    
    # Print which methods were evaluated
    print("\n" + "="*70)
    print("METHODS EVALUATED")
    print("="*70)
    print(f"\nSuccessfully evaluated {len(results)} methods:")
    for i, method in enumerate(results.keys(), 1):
        print(f"  {i}. {method}")
    print(f"\nTotal methods in results: {len(results)}")
    print()
    
    # Print summary statistics
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    # Convert EvaluationMetrics objects to dictionaries
    results_dict = {}
    print("Converting results to dictionaries...")
    for method_name, metrics in results.items():
        print(f"  Converting {method_name}: {type(metrics)}")
        if isinstance(metrics, EvaluationMetrics):
            results_dict[method_name] = metrics.get_metrics()
            print(f"    [OK] Converted {method_name} to metrics dict with {len(results_dict[method_name].get('localization_errors', []))} errors")
        else:
            # For domain gen metrics (if already a dict)
            results_dict[method_name] = metrics
            print(f"    [OK] Used {method_name} as dict")
    
    print(f"\n[OK] Total methods in results_dict: {len(results_dict)}")
    print(f"  Methods: {list(results_dict.keys())}\n")
    
    for method_name, metric_dict in results_dict.items():
        print(f"{method_name}:")
        if 'mean_error' in metric_dict:
            print(f"  Mean Localization Error: {metric_dict['mean_error']:.4f} m")
        if 'rmse_error' in metric_dict:
            print(f"  RMSE Localization Error: {metric_dict['rmse_error']:.4f} m")
        if 'error_90th' in metric_dict:
            print(f"  90th Percentile Error:   {metric_dict['error_90th']:.4f} m")
        if 'error_95th' in metric_dict:
            print(f"  95th Percentile Error:   {metric_dict['error_95th']:.4f} m")
        if 'std_error' in metric_dict:
            print(f"  Std Dev Error: {metric_dict['std_error']:.4f} m")
        if 'mean_reward' in metric_dict:
            print(f"  Mean Reward: {metric_dict['mean_reward']:.4f}")
        if 'total_steps' in metric_dict:
            print(f"  Total Steps: {metric_dict['total_steps']}")
        if 'battery_deviation' in metric_dict:
            print(f"  Battery Deviation: {metric_dict['battery_deviation']:.6f}")
        if 'los_ratio' in metric_dict:
            print(f"  LoS Selection Ratio: {metric_dict['los_ratio']:.2%}")
        
        # Print selection frequency top beacons (if available)
        if 'selection_frequency' in metric_dict:
            freq = metric_dict['selection_frequency']
            top_indices = np.argsort(freq)[::-1][:5]
            print(f"  Top 5 Selected Beacons: {', '.join([f'B{i}({freq[i]:.1%})' for i in top_indices])}")
        
        # Print generalization metrics (if available)
        if 'errors_by_beacon_count' in metric_dict:
            print(f"  Errors by Beacon Count:")
            for bc, err_stats in sorted(metric_dict['errors_by_beacon_count'].items()):
                print(f"    {bc} beacons: {err_stats['mean']:.4f} ± {err_stats['std']:.4f} m")
            print(f"  Errors by LoS Probability:")
            for los_p, err_stats in sorted(metric_dict['errors_by_los_prob'].items()):
                print(f"    LoS={los_p}: {err_stats['mean']:.4f} ± {err_stats['std']:.4f} m")
        
        print()
    
    # Show domain generalization comparison if available
    if 'Domain Gen DQN' in results_dict and 'DQN' in results_dict:
        print("="*70)
        print("DOMAIN GENERALIZATION IMPROVEMENT ANALYSIS")
        print("="*70 + "\n")
        
        dqn_metrics = results_dict['DQN']
        dgdqn_metrics = results_dict['Domain Gen DQN']
        
        error_improvement = (dqn_metrics['mean_error'] - dgdqn_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
        reward_improvement = (dgdqn_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
        
        print(f"DQN vs Domain Gen DQN (on same 100-epoch benchmark):")
        print(f"\n  Localization Error:")
        print(f"    Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m")
        print(f"    Domain Gen DQN:   {dgdqn_metrics['mean_error']:.4f} m ± {dgdqn_metrics['std_error']:.4f} m")
        print(f"    Improvement:      {error_improvement:+.2f}%")
        print(f"\n  Mean Reward:")
        print(f"    Standard DQN:     {dqn_metrics['mean_reward']:.4f}")
        print(f"    Domain Gen DQN:   {dgdqn_metrics['mean_reward']:.4f}")
        print(f"    Improvement:      {reward_improvement:+.2f}%")
        print(f"\n  90th Percentile Error:")
        print(f"    Standard DQN:     {dqn_metrics['error_90th']:.4f} m")
        print(f"    Domain Gen DQN:   {dgdqn_metrics['error_90th']:.4f} m")
        print()
    
    # Show meta-RL comparison if available
    if 'Meta RL' in results_dict and 'DQN' in results_dict:
        print("="*70)
        print("META-RL IMPROVEMENT ANALYSIS")
        print("="*70 + "\n")
        
        dqn_metrics = results_dict['DQN']
        meta_rl_metrics = results_dict['Meta RL']
        
        error_improvement = (dqn_metrics['mean_error'] - meta_rl_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
        reward_improvement = (meta_rl_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
        
        print(f"DQN vs Meta RL (on same 100-epoch benchmark):")
        print(f"\n  Localization Error:")
        print(f"    Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m")
        print(f"    Meta RL:          {meta_rl_metrics['mean_error']:.4f} m ± {meta_rl_metrics['std_error']:.4f} m")
        print(f"    Improvement:      {error_improvement:+.2f}%")
        print(f"\n  Mean Reward:")
        print(f"    Standard DQN:     {dqn_metrics['mean_reward']:.4f}")
        print(f"    Meta RL:          {meta_rl_metrics['mean_reward']:.4f}")
        print(f"    Improvement:      {reward_improvement:+.2f}%")
        print(f"\n  90th Percentile Error:")
        print(f"    Standard DQN:     {dqn_metrics['error_90th']:.4f} m")
        print(f"    Meta RL:          {meta_rl_metrics['error_90th']:.4f} m")
        print()
    
    # Show RL² LSTM comparison if available
    if 'RL² LSTM' in results_dict and 'DQN' in results_dict:
        print("="*70)
        print("RL² LSTM IMPROVEMENT ANALYSIS")
        print("="*70 + "\n")
        
        dqn_metrics = results_dict['DQN']
        rl2_lstm_metrics = results_dict['RL² LSTM']
        
        error_improvement = (dqn_metrics['mean_error'] - rl2_lstm_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
        reward_improvement = (rl2_lstm_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
        battery_dev_improvement = (dqn_metrics['battery_deviation'] - rl2_lstm_metrics['battery_deviation']) / dqn_metrics['battery_deviation'] * 100 if dqn_metrics['battery_deviation'] > 0 else 0
        
        print(f"DQN vs RL² LSTM (on same 100-epoch benchmark):")
        print(f"\n  Localization Error:")
        print(f"    Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['mean_error']:.4f} m ± {rl2_lstm_metrics['std_error']:.4f} m")
        print(f"    Improvement:      {error_improvement:+.2f}%")
        print(f"\n  Mean Reward:")
        print(f"    Standard DQN:     {dqn_metrics['mean_reward']:.4f}")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['mean_reward']:.4f}")
        print(f"    Improvement:      {reward_improvement:+.2f}%")
        print(f"\n  90th Percentile Error:")
        print(f"    Standard DQN:     {dqn_metrics['error_90th']:.4f} m")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['error_90th']:.4f} m")
        print(f"\n  95th Percentile Error:")
        print(f"    Standard DQN:     {dqn_metrics['error_95th']:.4f} m")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['error_95th']:.4f} m")
        print(f"\n  Battery Deviation (lower is better):")
        print(f"    Standard DQN:     {dqn_metrics['battery_deviation']:.6f}")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['battery_deviation']:.6f}")
        print(f"    Improvement:      {battery_dev_improvement:+.2f}%")
        print(f"\n  LoS Selection Ratio:")
        print(f"    Standard DQN:     {dqn_metrics['los_ratio']:.2%}")
        print(f"    RL² LSTM:         {rl2_lstm_metrics['los_ratio']:.2%}")
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

            # Add domain generalization comparison to file
            if 'Domain Gen DQN' in results_dict and 'DQN' in results_dict and method_name == 'Domain Gen DQN':
                f.write("="*50 + "\n")
                f.write("DOMAIN GENERALIZATION COMPARISON\n")
                f.write("="*50 + "\n\n")
                
                dqn_metrics = results_dict['DQN']
                dgdqn_metrics = results_dict['Domain Gen DQN']
                
                error_improvement = (dqn_metrics['mean_error'] - dgdqn_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
                reward_improvement = (dgdqn_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
                
                f.write("DQN vs Domain Gen DQN (on same 100-epoch benchmark):\n\n")
                f.write("Localization Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m\n")
                f.write(f"  Domain Gen DQN:   {dgdqn_metrics['mean_error']:.4f} m ± {dgdqn_metrics['std_error']:.4f} m\n")
                f.write(f"  Improvement:      {error_improvement:+.2f}%\n\n")
                
                f.write("Mean Reward:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_reward']:.4f}\n")
                f.write(f"  Domain Gen DQN:   {dgdqn_metrics['mean_reward']:.4f}\n")
                f.write(f"  Improvement:      {reward_improvement:+.2f}%\n\n")
                
                f.write("90th Percentile Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['error_90th']:.4f} m\n")
                f.write(f"  Domain Gen DQN:   {dgdqn_metrics['error_90th']:.4f} m\n")
            
            # Add meta-RL comparison to file
            if 'Meta RL' in results_dict and 'DQN' in results_dict and method_name == 'Meta RL':
                f.write("="*50 + "\n")
                f.write("META-RL COMPARISON\n")
                f.write("="*50 + "\n\n")
                
                dqn_metrics = results_dict['DQN']
                meta_rl_metrics = results_dict['Meta RL']
                
                error_improvement = (dqn_metrics['mean_error'] - meta_rl_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
                reward_improvement = (meta_rl_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
                
                f.write("DQN vs Meta RL (on same 100-epoch benchmark):\n\n")
                f.write("Localization Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m\n")
                f.write(f"  Meta RL:          {meta_rl_metrics['mean_error']:.4f} m ± {meta_rl_metrics['std_error']:.4f} m\n")
                f.write(f"  Improvement:      {error_improvement:+.2f}%\n\n")
                
                f.write("Mean Reward:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_reward']:.4f}\n")
                f.write(f"  Meta RL:          {meta_rl_metrics['mean_reward']:.4f}\n")
                f.write(f"  Improvement:      {reward_improvement:+.2f}%\n\n")
                
                f.write("90th Percentile Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['error_90th']:.4f} m\n")
                f.write(f"  Meta RL:          {meta_rl_metrics['error_90th']:.4f} m\n")
            
            # Add RL² LSTM comparison to file
            if 'RL² LSTM' in results_dict and 'DQN' in results_dict and method_name == 'RL² LSTM':
                f.write("="*50 + "\n")
                f.write("RL² LSTM COMPARISON\n")
                f.write("="*50 + "\n\n")
                
                dqn_metrics = results_dict['DQN']
                rl2_lstm_metrics = results_dict['RL² LSTM']
                
                error_improvement = (dqn_metrics['mean_error'] - rl2_lstm_metrics['mean_error']) / dqn_metrics['mean_error'] * 100
                reward_improvement = (rl2_lstm_metrics['mean_reward'] - dqn_metrics['mean_reward']) / abs(dqn_metrics['mean_reward']) * 100 if dqn_metrics['mean_reward'] != 0 else 0
                battery_dev_improvement = (dqn_metrics['battery_deviation'] - rl2_lstm_metrics['battery_deviation']) / dqn_metrics['battery_deviation'] * 100 if dqn_metrics['battery_deviation'] > 0 else 0
                
                f.write("DQN vs RL² LSTM (on same 100-epoch benchmark):\n\n")
                f.write("Localization Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_error']:.4f} m ± {dqn_metrics['std_error']:.4f} m\n")
                f.write(f"  RL² LSTM:         {rl2_lstm_metrics['mean_error']:.4f} m ± {rl2_lstm_metrics['std_error']:.4f} m\n")
                f.write(f"  Improvement:      {error_improvement:+.2f}%\n\n")
                
                f.write("Mean Reward:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['mean_reward']:.4f}\n")
                f.write(f"  RL² LSTM:         {rl2_lstm_metrics['mean_reward']:.4f}\n")
                f.write(f"  Improvement:      {reward_improvement:+.2f}%\n\n")
                
                f.write("90th Percentile Error:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['error_90th']:.4f} m\n")
                f.write(f"  RL² LSTM:         {rl2_lstm_metrics['error_90th']:.4f} m\n\n")
                
                f.write("Battery Deviation (lower is better):\n")
                f.write(f"  Standard DQN:     {dqn_metrics['battery_deviation']:.6f}\n")
                f.write(f"  RL² LSTM:         {rl2_lstm_metrics['battery_deviation']:.6f}\n")
                f.write(f"  Improvement:      {battery_dev_improvement:+.2f}%\n\n")
                
                f.write("LoS Selection Ratio:\n")
                f.write(f"  Standard DQN:     {dqn_metrics['los_ratio']:.2%}\n")
                f.write(f"  RL² LSTM:         {rl2_lstm_metrics['los_ratio']:.2%}\n")

    print(f"Summary saved to {summary_file}")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    print(f"Generating plots with {len(results_dict)} methods...")
    print(f"Methods available: {list(results_dict.keys())}")
    print()
    
    figs = {
        'ecdf': plot_ecdf_comparison(results_dict),
        'error': plot_error_comparison(results_dict),
        'battery': plot_battery_levels(results_dict),
        'deviation': plot_battery_deviation(results_dict),
        'lifetime': plot_infrastructure_lifetime(results_dict),
        'reward': plot_cumulative_reward(results_dict),
        'los_ratio': plot_los_ratio_comparison(results_dict),
        'movement': plot_agent_movement(results_dict),
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
        'movement': 'agent_movement.png',
    }
    
    for key, fig in figs.items():
        save_path = eval_dir / plot_names[key]
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    
    # Verification: Confirm which methods are in plots
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nMethods evaluated and plotted ({len(results_dict)} total):")
    for i, method_name in enumerate(results_dict.keys(), 1):
        marker = "[OK]" if method_name in ['RL² LSTM', 'Meta RL', 'Domain Gen DQN'] else " "
        print(f"  {marker} {i}. {method_name}")
    print()
    
    print(f"Results saved to: {eval_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()