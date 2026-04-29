"""
Network Lifetime Evaluation Module for UWB Beacon Selection System

This module evaluates network lifetime using 4 different metrics:
1. First Node Death (FND) - step when first beacon reaches 0% battery
2. Half Node Death (HND) - step when half of beacons are dead
3. Coverage Loss - step when fewer than NUM_SELECTED_BEACONS beacons are alive
4. Energy Threshold - step when any beacon reaches below threshold (default 10%)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from typing import Dict, List, Callable, Optional, Tuple

# Add src directory to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / 'src'))

from src.core.environment import Environment
from src.config import NUM_BEACONS, NUM_SELECTED_BEACONS, BEACON_INITIAL_BATTERY
from src.rl.trainer_dqn import DQNTrainer
from src.rl.train_meta_rl import MetaDQN
from src.rl.train_rl2_lstm import LSTM_DQN, STATE_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, POSSIBLE_ACTIONS, get_state
from src.localization.gdop import compute_weighted_gdop

# ============================================================================
# Core Lifetime Computation
# ============================================================================

def compute_lifetime(env: Environment, 
                     selection_fn: Callable, 
                     threshold: float = 10.0,
                     max_steps: int = 5000) -> Dict[str, Optional[int]]:
    """
    Compute network lifetime metrics for a single episode.
    
    Args:
        env: Environment instance
        selection_fn: Selection function that returns beacon indices
        threshold: Battery threshold percentage (default 10%)
        max_steps: Maximum simulation steps
    
    Returns:
        Dictionary with lifetime metrics:
        {
            "FND": first_node_death_step,
            "HND": half_node_death_step,
            "COVERAGE_LOSS": coverage_loss_step,
            "THRESHOLD": threshold_hit_step
        }
        Returns max_steps if metric not triggered.
    """
    # Reset environment for fresh episode
    env.reset_agent_to_random_location()
    env.reset_beacon_batteries()
    
    # Initialize metrics
    fnd = None  # First Node Death
    hnd = None  # Half Node Death
    coverage_loss = None  # Coverage Loss
    threshold_hit = None  # Energy Threshold
    
    initial_beacons = NUM_BEACONS
    half_beacons = initial_beacons / 2.0
    
    # Simulate steps
    for step in range(max_steps):
        # 1. Select beacons using the selection function
        selected = selection_fn(env)
        
        # 2. Apply selection to environment
        env.selected_beacon_indices = selected
        
        # 3. Step environment (moves agent, updates links, consumes energy)
        env.step()
        
        # 4. Get current battery levels
        batteries = np.array(env.get_battery_levels())
        
        # 5. Count alive beacons (battery > 0)
        alive = np.sum(batteries > 0)
        
        # 6. Check for threshold violations
        below_threshold = np.any(batteries < threshold)
        
        # 7. Record metrics on first trigger (never overwrite)
        if fnd is None and np.any(batteries <= 0):
            fnd = step
        
        if hnd is None and alive <= half_beacons:
            hnd = step
        
        if coverage_loss is None and alive < NUM_SELECTED_BEACONS:
            coverage_loss = step
        
        if threshold_hit is None and below_threshold:
            threshold_hit = step
        
        # 8. Termination: break if all metrics recorded
        if all(m is not None for m in [fnd, hnd, coverage_loss, threshold_hit]):
            break
    
    # Replace None with max_steps if not triggered
    return {
        "FND": fnd if fnd is not None else max_steps,
        "HND": hnd if hnd is not None else max_steps,
        "COVERAGE_LOSS": coverage_loss if coverage_loss is not None else max_steps,
        "THRESHOLD": threshold_hit if threshold_hit is not None else max_steps
    }


# ============================================================================
# Multi-run Evaluation
# ============================================================================

def evaluate_method(selection_fn: Callable, 
                    num_runs: int = 100,
                    method_name: str = "Method",
                    seed_base: int = 0,
                    wrapper: Optional['RL2LSTMWrapper'] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a selection method across multiple runs.
    
    Args:
        selection_fn: Selection function that takes env and returns beacon indices
        num_runs: Number of simulation runs (default 100)
        method_name: Name of the method for progress bar
        seed_base: Base seed for reproducibility
        wrapper: Optional RL2LSTMWrapper to reset hidden state per run
    
    Returns:
        Dictionary with statistics for each metric:
        {
            "FND": {"mean": x, "std": x, "min": x, "max": x},
            "HND": {"mean": x, "std": x, "min": x, "max": x},
            "COVERAGE_LOSS": {"mean": x, "std": x, "min": x, "max": x},
            "THRESHOLD": {"mean": x, "std": x, "min": x, "max": x}
        }
    """
    # Collect results for each metric
    fnd_values = []
    hnd_values = []
    coverage_loss_values = []
    threshold_values = []
    
    pbar = tqdm(range(num_runs), desc=f"Evaluating {method_name}", leave=True)
    
    for run in pbar:
        # Set seed for reproducibility
        np.random.seed(seed_base + run)
        torch.manual_seed(seed_base + run)
        
        # Reset hidden state for RL2 LSTM if wrapper is provided
        if wrapper is not None:
            wrapper.reset_hidden_state()
        
        # Create new environment for this run
        env = Environment()
        
        # Compute lifetime metrics for this run
        metrics = compute_lifetime(env, selection_fn)
        
        # Collect results
        fnd_values.append(metrics["FND"])
        hnd_values.append(metrics["HND"])
        coverage_loss_values.append(metrics["COVERAGE_LOSS"])
        threshold_values.append(metrics["THRESHOLD"])
    
    # Compute statistics for each metric
    def compute_stats(values: List[int]) -> Dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values))
        }
    
    return {
        "FND": compute_stats(fnd_values),
        "HND": compute_stats(hnd_values),
        "COVERAGE_LOSS": compute_stats(coverage_loss_values),
        "THRESHOLD": compute_stats(threshold_values)
    }


# ============================================================================
# Selection Methods
# ============================================================================

def random_selection(env: Environment) -> List[int]:
    """Random beacon selection."""
    return list(np.random.choice(NUM_BEACONS, NUM_SELECTED_BEACONS, replace=False))


def nearest_neighbor_selection(env: Environment) -> List[int]:
    """Nearest neighbor beacon selection."""
    agent_pos = np.array(env.agent.get_position())
    beacon_positions = np.array([beacon.position for beacon in env.beacons])
    distances = np.linalg.norm(beacon_positions - agent_pos, axis=1)
    return list(np.argsort(distances)[:NUM_SELECTED_BEACONS])


def gdop_selection(env: Environment) -> List[int]:
    """GDOP-optimized beacon selection using weighted GDOP."""
    agent_pos = np.array(env.agent.get_position())
    best_score = float('inf')
    best_combo = None
    
    # Try all combinations of NUM_SELECTED_BEACONS beacons
    for combo in combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS):
        selected_positions = [env.beacons[i].position for i in combo]
        selected_los_flags = [env.current_links[i] for i in combo]
        
        score = compute_weighted_gdop(
            agent_estimate=agent_pos,
            beacon_positions=selected_positions,
            los_flags=selected_los_flags
        )
        
        if score < best_score:
            best_score = score
            best_combo = combo
    
    return list(best_combo) if best_combo is not None else list(range(NUM_SELECTED_BEACONS))


def dqn_selection(env: Environment, trainer: DQNTrainer) -> List[int]:
    """DQN-based beacon selection."""
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


def domain_gen_selection(env: Environment, trainer: DQNTrainer) -> List[int]:
    """Domain-generalized DQN beacon selection."""
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


def meta_rl_selection(env: Environment, meta_model: nn.Module) -> List[int]:
    """Meta-RL beacon selection."""
    device = next(meta_model.parameters()).device
    state = torch.tensor(np.array(env.get_battery_levels(), dtype=np.float32)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = meta_model(state)
        action_idx = q_values.argmax(dim=1).item()
    
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    return list(possible_actions[action_idx])


class RL2LSTMWrapper:
    """Wrapper for RL2 LSTM model with hidden state management."""
    
    def __init__(self, model: LSTM_DQN, device: torch.device):
        self.model = model
        self.device = device
        self.hidden_state = None
        self.possible_actions = POSSIBLE_ACTIONS
    
    def reset_hidden_state(self):
        """Reset hidden state for new episode."""
        self.hidden_state = None
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using LSTM-DQN with persistent hidden state."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values, self.hidden_state = self.model(state_tensor, self.hidden_state)
            action = q_values.argmax(dim=-1).item()
        return action


def rl2_selection(env: Environment, wrapper: RL2LSTMWrapper) -> List[int]:
    """RL2 LSTM beacon selection."""
    state = get_state(env)
    action_idx = wrapper.select_action(state)
    return list(wrapper.possible_actions[action_idx])


# ============================================================================
# Model Loading
# ============================================================================

def load_dqn_model(model_path: Optional[Path] = None) -> Optional[DQNTrainer]:
    """Load DQN model from checkpoint."""
    if model_path is None:
        model_path = Path(__file__).parent / 'models' / 'dqn_env_0.pt'
    
    if model_path.exists():
        trainer = DQNTrainer(state_size=NUM_BEACONS)
        trainer.load_model(str(model_path))
        trainer.epsilon = 0.0
        print(f"  [OK] DQN model loaded from {model_path.name}")
        return trainer
    else:
        print(f"  [FAIL] DQN model not found at {model_path}")
        return None


def load_domain_gen_model() -> Optional[DQNTrainer]:
    """Load domain-generalized DQN model."""
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / 'domain_generalization'
    
    if checkpoint_dir.exists():
        model_files = sorted(checkpoint_dir.glob('dqn_domain_generalization_*.pt'), reverse=True)
        if model_files:
            model_path = model_files[0]
            trainer = DQNTrainer(state_size=NUM_BEACONS)
            trainer.load_model(str(model_path))
            trainer.epsilon = 0.0
            print(f"  [OK] Domain Gen DQN model loaded from {model_path.name}")
            return trainer
    
    print(f"  [FAIL] Domain Gen DQN model not found")
    return None


def load_meta_rl_model() -> Optional[nn.Module]:
    """Load meta-RL model."""
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / 'meta_rl'
    
    if checkpoint_dir.exists():
        meta_model_files = sorted(checkpoint_dir.glob('meta_dqn_*_*.pt'), reverse=True)
        if meta_model_files:
            model_path = meta_model_files[0]
            
            try:
                action_size = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))
                meta_model = MetaDQN(state_size=NUM_BEACONS, action_size=action_size, hidden_size=64)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                meta_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                meta_model.to(device)
                meta_model.eval()
                print(f"  [OK] Meta-RL model loaded from {model_path.name}")
                return meta_model
            except Exception as e:
                print(f"  [FAIL] Error loading Meta-RL model: {e}")
                return None
    
    print(f"  [FAIL] Meta-RL model not found")
    return None


def load_rl2_lstm_model() -> Optional[RL2LSTMWrapper]:
    """Load RL2 LSTM model."""
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / 'rl2_lstm'
    
    if checkpoint_dir.exists():
        rl2_model_files = sorted(checkpoint_dir.glob('rl2_lstm_*_*.pt'), reverse=True)
        if rl2_model_files:
            model_path = rl2_model_files[0]
            
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                rl2_lstm_model = LSTM_DQN(
                    input_size=STATE_SIZE,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    num_actions=len(POSSIBLE_ACTIONS),
                    num_layers=LSTM_NUM_LAYERS
                ).to(device)
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                rl2_lstm_model.load_state_dict(checkpoint['model_state_dict'])
                rl2_lstm_model.eval()
                
                wrapper = RL2LSTMWrapper(rl2_lstm_model, device)
                print(f"  [OK] RL2 LSTM model loaded from {model_path.name}")
                return wrapper
            except Exception as e:
                print(f"  [FAIL] Error loading RL2 LSTM model: {e}")
                return None
    
    print(f"  [FAIL] RL2 LSTM model not found")
    return None


# ============================================================================
# Output Formatting
# ============================================================================

def print_results_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print results in clean table format."""
    print("\n" + "="*100)
    print("NETWORK LIFETIME EVALUATION RESULTS")
    print("="*100 + "\n")
    
    metrics = ["FND", "HND", "COVERAGE_LOSS", "THRESHOLD"]
    stats = ["mean", "std", "min", "max"]
    
    # Print header for each metric
    for metric in metrics:
        print(f"\n{metric} (First {metric == 'FND' and 'Node Death' or metric == 'HND' and 'Half Node Death' or metric == 'COVERAGE_LOSS' and 'Coverage Loss' or 'Energy Threshold Exceeded'}):")
        print("-" * 100)
        print(f"{'Method':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
        print("-" * 100)
        
        for method_name in sorted(results.keys()):
            if metric in results[method_name]:
                metric_data = results[method_name][metric]
                print(f"{method_name:<20} {metric_data['mean']:<15.2f} {metric_data['std']:<15.2f} "
                      f"{metric_data['min']:<15} {metric_data['max']:<15}")
    
    print("\n" + "="*100)


def save_results_json(results: Dict[str, Dict[str, Dict[str, float]]], 
                      output_file: Path = None) -> None:
    """Save results to JSON file."""
    if output_file is None:
        output_file = Path(__file__).parent / 'results' / 'network_lifetime.json'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")


def plot_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Generate bar plots for lifetime metrics."""
    metrics = ["FND", "HND", "COVERAGE_LOSS", "THRESHOLD"]
    metric_labels = {
        "FND": "First Node Death",
        "HND": "Half Node Death",
        "COVERAGE_LOSS": "Coverage Loss",
        "THRESHOLD": "Energy Threshold"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        methods = list(results.keys())
        means = [results[m][metric]["mean"] for m in methods]
        stds = [results[m][metric]["std"] for m in methods]
        
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Selection Method', fontsize=11, fontweight='bold')
        ax.set_ylabel('Steps', fontsize=11, fontweight='bold')
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(__file__).parent / 'results' / 'network_lifetime_plots.png'
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Plots saved to {plot_file}")
    plt.close()


# ============================================================================
# Main Evaluation
# ============================================================================

def main(num_runs: int = 100, plot: bool = True):
    """
    Run comprehensive network lifetime evaluation for all selection methods.
    
    Args:
        num_runs: Number of simulation runs per method (default 100)
        plot: Whether to generate plots (default True)
    """
    print("\n" + "="*100)
    print("NETWORK LIFETIME EVALUATION MODULE")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Number of Beacons: {NUM_BEACONS}")
    print(f"  Selected per Step: {NUM_SELECTED_BEACONS}")
    print(f"  Runs per Method: {num_runs}")
    print(f"  Energy Threshold: 10%")
    print("\n" + "="*100)
    
    results = {}
    seed_base = 42
    
    # ========================================================================
    # 1. Random Selection
    # ========================================================================
    print("\n1. Evaluating Random Selection...")
    results['Random'] = evaluate_method(
        random_selection, 
        num_runs=num_runs, 
        method_name='Random',
        seed_base=seed_base
    )
    
    # ========================================================================
    # 2. Nearest Neighbor Selection
    # ========================================================================
    print("\n2. Evaluating Nearest Neighbor Selection...")
    results['Nearest Neighbor'] = evaluate_method(
        nearest_neighbor_selection, 
        num_runs=num_runs, 
        method_name='Nearest Neighbor',
        seed_base=seed_base
    )
    
    # ========================================================================
    # 3. GDOP Selection
    # ========================================================================
    print("\n3. Evaluating GDOP Selection...")
    results['GDOP'] = evaluate_method(
        gdop_selection,
        num_runs=num_runs,
        method_name='GDOP',
        seed_base=seed_base
    )
    
    # ========================================================================
    # 4. DQN Selection
    # ========================================================================
    print("\n4. Loading DQN model...")
    dqn_model = load_dqn_model()
    if dqn_model is not None:
        print("   Evaluating DQN Selection...")
        results['DQN'] = evaluate_method(
            lambda env: dqn_selection(env, dqn_model),
            num_runs=num_runs,
            method_name='DQN',
            seed_base=seed_base
        )
    else:
        print("   [WARN] Skipping DQN evaluation - model not available")
    
    # ========================================================================
    # 5. Domain Generalization Selection
    # ========================================================================
    print("\n5. Loading Domain Generalization model...")
    domain_gen_model = load_domain_gen_model()
    if domain_gen_model is not None:
        print("   Evaluating Domain Gen Selection...")
        results['Domain Gen'] = evaluate_method(
            lambda env: domain_gen_selection(env, domain_gen_model),
            num_runs=num_runs,
            method_name='Domain Gen',
            seed_base=seed_base
        )
    else:
        print("   [WARN] Skipping Domain Gen evaluation - model not available")
    
    # ========================================================================
    # 6. Meta-RL Selection
    # ========================================================================
    print("\n6. Loading Meta-RL model...")
    meta_rl_model = load_meta_rl_model()
    if meta_rl_model is not None:
        print("   Evaluating Meta-RL Selection...")
        results['Meta-RL'] = evaluate_method(
            lambda env: meta_rl_selection(env, meta_rl_model),
            num_runs=num_runs,
            method_name='Meta-RL',
            seed_base=seed_base
        )
    else:
        print("   [WARN] Skipping Meta-RL evaluation - model not available")
    
    # ========================================================================
    # 7. RL2 LSTM Selection
    # ========================================================================
    print("\n7. Loading RL2 LSTM model...")
    rl2_wrapper = load_rl2_lstm_model()
    if rl2_wrapper is not None:
        print("   Evaluating RL2 LSTM Selection...")
        
        # Create wrapper function for RL2 LSTM
        def rl2_selection_fn(env):
            return rl2_selection(env, rl2_wrapper)
        
        results['RL2 LSTM'] = evaluate_method(
            rl2_selection_fn,
            num_runs=num_runs,
            method_name='RL2 LSTM',
            seed_base=seed_base,
            wrapper=rl2_wrapper
        )
    else:
        print("   [WARN] Skipping RL2 LSTM evaluation - model not available")
    
    # ========================================================================
    # Output Results
    # ========================================================================
    print_results_table(results)
    save_results_json(results)
    
    if plot:
        print("\nGenerating plots...")
        plot_results(results)
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Network Lifetime Evaluation')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per method (default: 100)')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate plots')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Skip plot generation')
    
    args = parser.parse_args()
    
    results = main(num_runs=args.runs, plot=args.plot)
