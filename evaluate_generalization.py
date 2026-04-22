"""
Generalization Evaluation for UWB RL Models.

Evaluates all trained models on unseen test environments and produces:
  1. results/generalization_results.json  — full metrics
    2. results/per_test_environment/*.json  — one file per test env
    3. Console summary tables
    4. Plots:
         - generalization_ecdf.png
         - generalization_mean_error.png
         - generalization_boxplot.png

Usage:
    python evaluate_generalization.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'rl'))

from config import NUM_BEACONS, NUM_SELECTED_BEACONS
from rl.cir_training_config import setup_cir_training, FAST_TRAINING
from rl.trainer_dqn import DQNTrainer
from rl.train_domain_generalization import create_env_from_config
from rl.train_meta_rl import MetaDQN, META_STATE_SIZE, build_meta_state
from rl.train_rl2_lstm import (
    LSTM_DQN, STATE_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    ACTION_SIZE, POSSIBLE_ACTIONS, get_state as rl2_get_state,
)
from reward.reward import compute_reward
from localization.trilateration import trilateration_2d, compute_noisy_distances

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POSSIBLE_BEACON_COMBOS = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
NUM_TRAIN_ENVIRONMENTS = 40
MIN_TRIANGLE_AREA = 0.5


def set_global_seeds(seed: int) -> None:
    """Set seeds for deterministic evaluation runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Evaluation core
# ============================================================

def evaluate_on_environment(
    config: dict,
    selection_fn,
    max_steps: int = 2000,
    num_episodes: int = 10,
) -> dict:
    """
    Evaluate a beacon-selection function on a single environment config.

    Runs *num_episodes* episodes, each starting with fresh batteries and
    a random agent position.  Collects localization errors, rewards,
    battery deviation, and network lifetime.

    Args:
        config: Environment configuration dict (from env_configs.json).
        selection_fn: Callable(env) -> list[int] returning beacon indices.
                      May have a `.reset()` method (called before each episode).
        max_steps: Maximum steps per episode.
        num_episodes: Number of independent episodes per environment.

    Returns:
        Metrics dict with keys: mean_error, rmse, p90, p95, std_error,
        battery_deviation, network_lifetime, mean_reward, all_errors.
    """
    all_errors: list[float] = []
    all_rewards: list[float] = []
    final_batteries_list: list[np.ndarray] = []
    lifetimes: list[int] = []

    base_seed = int(config.get('seed', config.get('env_id', 0)))

    for _ep in range(num_episodes):
        set_global_seeds(base_seed + _ep)
        # Reset hidden state (RL² LSTM) or other per-episode state
        if hasattr(selection_fn, 'reset'):
            selection_fn.reset()

        env = create_env_from_config(config)
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()

        lifetime = max_steps

        for step in range(max_steps):
            # --- Select beacons ---
            selected_indices = selection_fn(env)

            # --- Apply action and step ---
            env.selected_beacon_indices = selected_indices
            env.step()

            # --- Compute localization error with geometry and numerical safeguards ---
            agent_pos = np.array(env.agent.get_position())
            sel_pos = np.array([env.beacons[i].position for i in selected_indices])
            los_flags = [env.current_links[i] for i in selected_indices]
            battery_levels = env.get_battery_levels()

            geometry_valid = True
            if len(sel_pos) >= 3:
                area = abs(
                    0.5 * (
                        sel_pos[0, 0] * (sel_pos[1, 1] - sel_pos[2, 1]) +
                        sel_pos[1, 0] * (sel_pos[2, 1] - sel_pos[0, 1]) +
                        sel_pos[2, 0] * (sel_pos[0, 1] - sel_pos[1, 1])
                    )
                )
                geometry_valid = area >= MIN_TRIANGLE_AREA

            if not geometry_valid:
                error = 50.0
            else:
                try:
                    distances = compute_noisy_distances(agent_pos, sel_pos, los_flags)
                    est_x, est_y = trilateration_2d(sel_pos, distances)

                    if not np.isfinite(est_x) or not np.isfinite(est_y):
                        raise ValueError("Invalid estimate")

                    est_pos = np.array([est_x, est_y])
                    error = float(np.linalg.norm(agent_pos - est_pos))

                    # Clip unrealistic failures
                    error = min(error, 50.0)
                except Exception:
                    error = 50.0

            # Normalize error by environment scale for cross-domain comparability.
            scale = max(float(config['grid_width']), float(config['grid_height']), 1.0)
            error = error / scale
            all_errors.append(error)

            reward = compute_reward(agent_pos, sel_pos, los_flags, battery_levels)
            all_rewards.append(float(reward))

            # --- Battery depletion check ---
            if min(battery_levels) <= 10.0:
                lifetime = step + 1
                break

        lifetimes.append(lifetime)
        final_batteries_list.append(np.array(env.get_battery_levels()))

    errors_arr = np.array(all_errors)

    # Battery deviation: MSE of final battery levels around their mean
    avg_final = np.mean(final_batteries_list, axis=0)
    mean_batt = np.mean(avg_final)
    batt_dev = float(np.mean(((avg_final - mean_batt) / (mean_batt + 1e-6)) ** 2))

    return {
        'mean_error': float(np.mean(errors_arr)),
        'rmse': float(np.sqrt(np.mean(errors_arr ** 2))),
        'p90': float(np.percentile(errors_arr, 90)),
        'p95': float(np.percentile(errors_arr, 95)),
        'std_error': float(np.std(errors_arr)),
        'battery_deviation': batt_dev,
        'network_lifetime': float(np.mean(lifetimes)),
        'mean_reward': float(np.mean(all_rewards)),
        'all_errors': errors_arr.tolist(),
    }


# ============================================================
# Selection-function factories
# ============================================================

def make_dqn_selection(trainer: DQNTrainer):
    """Wrap a DQNTrainer in a selection callable."""
    def _select(env):
        state = trainer.state_to_vector(env)
        action = trainer.select_action(state, training=False)
        return list(trainer.possible_actions[action])
    return _select


def make_avg_dqn_selection(trainers: list):
    """
    Average Q-values from multiple per-environment DQN models.
    Returns the action with the highest *average* Q-value.
    """
    def _select(env):
        action_votes = []
        for tr in trainers:
            state = tr.state_to_vector(env)
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(tr.device)
                q = tr.q_network(st)
                action_votes.append(int(q.argmax(dim=1).item()))

        action = Counter(action_votes).most_common(1)[0][0]
        return list(trainers[0].possible_actions[action])
    return _select


def make_meta_rl_selection(meta_model):
    """Wrap MetaDQN in a selection callable."""
    device = next(meta_model.parameters()).device

    def _select(env):
        if getattr(meta_model, 'state_size', META_STATE_SIZE) == NUM_BEACONS:
            state_vec = np.array(env.get_battery_levels(), dtype=np.float32) / 100.0
        else:
            state_vec = build_meta_state(env)
        state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = meta_model(state)
            action_idx = q_values.argmax(dim=1).item()
        return list(POSSIBLE_BEACON_COMBOS[action_idx])
    return _select


def make_rl2_lstm_selection(model, device):
    """
    Wrap LSTM_DQN in a selection callable that maintains hidden state
    across steps within an episode and exposes a `.reset()` to clear it
    between episodes.
    """
    hidden_state = [None]  # mutable container

    def _select(env):
        state = rl2_get_state(env)
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values, hidden_state[0] = model(st, hidden_state[0])
            action = q_values.argmax(dim=-1).item()
        return list(POSSIBLE_ACTIONS[action])

    def _reset():
        hidden_state[0] = None

    _select.reset = _reset
    return _select


# ============================================================
# Model loaders
# ============================================================

def load_dqn_models(model_dir: Path, num_envs: int = NUM_TRAIN_ENVIRONMENTS) -> list:
    """Load all per-environment DQN models."""
    trainers = []
    for i in range(num_envs):
        p = model_dir / f'dqn_env_{i}.pt'
        if p.exists():
            tr = DQNTrainer(state_size=NUM_BEACONS)
            tr.load_model(str(p))
            tr.epsilon = 0.0
            trainers.append(tr)
        else:
            print(f"  [WARN] {p.name} not found — skipping")
    return trainers


def load_domain_gen_model(model_dir: Path):
    p = model_dir / 'dqn_domain_gen.pt'
    if not p.exists():
        print(f"  [WARN] {p.name} not found")
        return None
    tr = DQNTrainer(state_size=NUM_BEACONS)
    tr.load_model(str(p))
    tr.epsilon = 0.0
    return tr


def load_meta_rl_model(model_dir: Path):
    p = model_dir / 'meta_rl.pt'
    if not p.exists():
        print(f"  [WARN] {p.name} not found")
        return None
    action_size = len(POSSIBLE_BEACON_COMBOS)
    state_dict = torch.load(str(p), map_location=DEVICE, weights_only=False)

    # Primary path: new normalized meta-state.
    try:
        model = MetaDQN(state_size=META_STATE_SIZE, action_size=action_size, hidden_size=64)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception:
        # Backward-compatible fallback for legacy checkpoints.
        model = MetaDQN(state_size=NUM_BEACONS, action_size=action_size, hidden_size=64)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model


def load_rl2_lstm_model(model_dir: Path):
    p = model_dir / 'rl2_lstm.pt'
    if not p.exists():
        print(f"  [WARN] {p.name} not found")
        return None
    model = LSTM_DQN(
        input_size=STATE_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_actions=ACTION_SIZE,
        num_layers=LSTM_NUM_LAYERS,
    ).to(DEVICE)
    ckpt = torch.load(str(p), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ============================================================
# Plotting
# ============================================================

def plot_ecdf(results: dict, save_path: Path):
    """ECDF comparison of localization errors across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for idx, (method, env_results) in enumerate(results.items()):
        all_errors = []
        for er in env_results.values():
            all_errors.extend(er['all_errors'])
        errors = np.sort(all_errors)
        ecdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, ecdf, linewidth=2.5, label=method,
                color=colors[idx % len(colors)])

    ax.axhline(0.90, ls='--', color='gray', alpha=0.5)
    ax.axhline(0.95, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('Normalized Localization Error', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('ECDF — Generalization Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mean_error_bars(summary: dict, save_path: Path):
    """Bar plot of mean localization error with std-dev error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(summary.keys())
    means = [summary[m]['mean_error'] for m in methods]
    stds = [summary[m]['std_error'] for m in methods]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    bars = ax.bar(methods, means, yerr=stds, capsize=8, alpha=0.85,
                  color=colors, edgecolor='black', linewidth=0.5)
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Mean Normalized Localization Error', fontsize=13)
    ax.set_title('Mean Error — Unseen Environments', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_boxplot(results: dict, save_path: Path):
    """Box plot of error distributions across test environments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(results.keys())
    data = []
    for method in methods:
        all_errors = []
        for er in results[method].values():
            all_errors.extend(er['all_errors'])
        data.append(all_errors)

    bp = ax.boxplot(data, labels=methods, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)

    ax.set_ylabel('Normalized Localization Error', fontsize=13)
    ax.set_title('Error Distribution — Unseen Environments',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_per_test_environment_results(
    test_configs: list,
    all_results: dict,
    methods: list,
    output_dir: Path,
) -> dict:
    """Save one JSON file per test environment and return env-centric metrics."""
    output_dir.mkdir(exist_ok=True)
    per_test_environment: dict = {}

    for env_idx, config in enumerate(test_configs):
        env_key = f'env_{env_idx}'
        env_payload = {
            'env_key': env_key,
            'env_id': config['env_id'],
            'grid_width': config['grid_width'],
            'grid_height': config['grid_height'],
            'los_probability': config['los_probability'],
            'metrics_by_method': {},
        }

        for method_name in methods:
            metrics = all_results[method_name][env_key]
            env_payload['metrics_by_method'][method_name] = {
                k: v for k, v in metrics.items() if k != 'all_errors'
            }

        per_test_environment[env_key] = env_payload
        out_path = output_dir / f"test_env_{env_idx}_id_{config['env_id']}.json"
        with open(out_path, 'w') as f:
            json.dump(env_payload, f, indent=2)

    return per_test_environment


# ============================================================
# Main
# ============================================================

def main():
    set_global_seeds(42)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"  GENERALIZATION EVALUATION  —  {ts}")
    print("=" * 70)

    setup_cir_training(FAST_TRAINING)

    # ------------------------------------------------------------------
    # Load environment configs
    # ------------------------------------------------------------------
    configs_path = ROOT / 'data' / 'env_configs.json'
    if not configs_path.exists():
        print(f"\nERROR: {configs_path} not found.  Run `python run_experiment.py` first.")
        return

    with open(configs_path, 'r') as f:
        all_configs = json.load(f)

    train_configs_path = ROOT / 'data' / 'train_configs.json'
    test_configs_path = ROOT / 'data' / 'test_configs.json'

    if test_configs_path.exists():
        with open(test_configs_path, 'r') as f:
            test_configs = json.load(f)
        if train_configs_path.exists():
            with open(train_configs_path, 'r') as f:
                train_configs = json.load(f)
        else:
            train_configs = all_configs[:NUM_TRAIN_ENVIRONMENTS]
    else:
        train_configs = all_configs[:NUM_TRAIN_ENVIRONMENTS]
        test_configs = all_configs[NUM_TRAIN_ENVIRONMENTS:]

    print(f"\nTest environments: {len(test_configs)}")
    for c in test_configs:
        print(f"  env_id={c['env_id']}, grid={c['grid_width']}x{c['grid_height']}, "
              f"LoS={c['los_probability']:.2f}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    model_dir = ROOT / 'models'
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)

    print(f"\nLoading models from {model_dir} ...")

    dqn_trainers = load_dqn_models(model_dir, num_envs=len(train_configs))
    domain_gen = load_domain_gen_model(model_dir)
    meta_model = load_meta_rl_model(model_dir)
    rl2_model = load_rl2_lstm_model(model_dir)

    print(f"  DQN per-env   : {len(dqn_trainers)} models")
    print(f"  Domain Gen DQN: {'OK' if domain_gen else 'NOT FOUND'}")
    print(f"  Meta RL       : {'OK' if meta_model else 'NOT FOUND'}")
    print(f"  RL² LSTM      : {'OK' if rl2_model else 'NOT FOUND'}")

    # Build methods dict (only include models that loaded successfully)
    methods: dict = {}
    if dqn_trainers:
        methods[f'DQN (Avg {len(dqn_trainers)})'] = make_avg_dqn_selection(dqn_trainers)
    if domain_gen:
        methods['Domain Gen DQN'] = make_dqn_selection(domain_gen)
    if meta_model:
        methods['Meta RL'] = make_meta_rl_selection(meta_model)
    if rl2_model:
        methods['RL² LSTM'] = make_rl2_lstm_selection(rl2_model, DEVICE)

    if not methods:
        print("\nERROR: No models found.  Run `python run_experiment.py` first.")
        return

    # ------------------------------------------------------------------
    # Evaluate on each test environment
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"EVALUATING {len(methods)} METHODS ON {len(test_configs)} UNSEEN ENVIRONMENTS")
    print(f"{'=' * 70}")

    # {method_name: {env_key: metrics_dict}}
    all_results: dict = {m: {} for m in methods}

    for env_idx, config in enumerate(test_configs):
        print(f"\n--- Test Environment {env_idx}  "
              f"(env_id={config['env_id']}, "
              f"grid={config['grid_width']}x{config['grid_height']}, "
              f"LoS={config['los_probability']:.2f}) ---")

        for method_name, select_fn in methods.items():
            metrics = evaluate_on_environment(
                config, select_fn, max_steps=2000, num_episodes=10,
            )
            all_results[method_name][f'env_{env_idx}'] = metrics

            print(f"  {method_name:20s} | "
                f"Mean={metrics['mean_error']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  "
                f"P90={metrics['p90']:.4f}  "
                f"Life={metrics['network_lifetime']:.0f}")

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE  (averaged over all test environments)")
    print(f"{'=' * 70}\n")

    summary: dict = {}
    col = f"{'Method':20s} | {'Mean Err':>9s} | {'RMSE':>9s} | {'P90':>9s} | {'P95':>9s} | {'Std':>9s} | {'Batt Dev':>9s} | {'Lifetime':>9s}"
    print(col)
    print("-" * len(col))

    for method_name in methods:
        env_m = all_results[method_name]
        s = {
            'mean_error':        float(np.mean([v['mean_error']        for v in env_m.values()])),
            'rmse':              float(np.mean([v['rmse']              for v in env_m.values()])),
            'p90':               float(np.mean([v['p90']               for v in env_m.values()])),
            'p95':               float(np.mean([v['p95']               for v in env_m.values()])),
            'std_error':         float(np.mean([v['std_error']         for v in env_m.values()])),
            'battery_deviation': float(np.mean([v['battery_deviation'] for v in env_m.values()])),
            'network_lifetime':  float(np.mean([v['network_lifetime']  for v in env_m.values()])),
        }
        summary[method_name] = s
        print(f"{method_name:20s} | {s['mean_error']:9.4f} | {s['rmse']:9.4f} | "
              f"{s['p90']:9.4f} | {s['p95']:9.4f} | {s['std_error']:9.4f} | "
              f"{s['battery_deviation']:9.6f} | {s['network_lifetime']:9.1f}")

    print(f"\n{'=' * 70}")
    print('PER-TEST-ENVIRONMENT RESULTS')
    print(f"{'=' * 70}")
    for env_idx, config in enumerate(test_configs):
        env_key = f'env_{env_idx}'
        print(
            f"\nTest env {env_idx} (env_id={config['env_id']}, "
            f"grid={config['grid_width']}x{config['grid_height']}, "
            f"LoS={config['los_probability']:.2f})"
        )
        for method_name in methods:
            m = all_results[method_name][env_key]
            print(
                f"  {method_name:20s} | Mean={m['mean_error']:.4f} "
                f"RMSE={m['rmse']:.4f} P95={m['p95']:.4f} "
                f"Life={m['network_lifetime']:.0f}"
            )

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    per_env_dir = results_dir / 'per_test_environment'
    per_test_environment = save_per_test_environment_results(
        test_configs=test_configs,
        all_results=all_results,
        methods=list(methods.keys()),
        output_dir=per_env_dir,
    )

    output = {
        'timestamp': datetime.now().isoformat(),
        'num_train_envs': len(train_configs),
        'num_test_envs': len(test_configs),
        'train_env_ids': [c['env_id'] for c in train_configs],
        'test_env_ids': [c['env_id'] for c in test_configs],
        'methods': list(methods.keys()),
        'summary': summary,
        'per_environment': {},
        'per_test_environment': per_test_environment,
    }

    for method_name in methods:
        output['per_environment'][method_name] = {}
        for env_key, env_metrics in all_results[method_name].items():
            # Omit large 'all_errors' array from JSON to save space
            output['per_environment'][method_name][env_key] = {
                k: v for k, v in env_metrics.items() if k != 'all_errors'
            }

    results_path = results_dir / 'generalization_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {results_path}")
    print(f"Per-test-env JSON files → {per_env_dir}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("GENERATING PLOTS")
    print(f"{'=' * 70}")

    plot_ecdf(all_results, results_dir / 'generalization_ecdf.png')
    plot_mean_error_bars(summary, results_dir / 'generalization_mean_error.png')
    plot_boxplot(all_results, results_dir / 'generalization_boxplot.png')

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Results JSON : {results_path.resolve()}")
    print(f"  Plots        : {results_dir.resolve()}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
