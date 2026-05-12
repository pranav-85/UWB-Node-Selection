"""
Experiment Pipeline Orchestrator for UWB RL Models.

Generates 50 randomized environments, splits into 40 train / 10 test,
and trains all model types using EXISTING training scripts.

No training logic is duplicated — all training is delegated to:
  - src/rl/trainer_dqn.py  (DQNTrainer)
  - src/rl/train_domain_generalization.py  (train_across_environments)
  - src/rl/train_meta_rl.py  (train_meta_rl)
  - src/rl/train_rl2_lstm.py  (train_rl2)

Usage:
    python run_experiment.py
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup: add both src/ and src/rl/ so that internal sibling imports
# (e.g. `from trainer_dqn import DQNTrainer` inside train_domain_generalization)
# resolve correctly.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'rl'))

from config import NUM_BEACONS, NUM_SELECTED_BEACONS
from rl.cir_training_config import setup_cir_training, FAST_TRAINING
from rl.trainer_dqn import DQNTrainer
from rl.train_domain_generalization import (
    create_env_from_config,
    train_across_environments,
)
from rl.train_meta_rl import MetaDQN, train_meta_rl, META_STATE_SIZE
from rl.train_rl2_lstm import train_rl2


TOTAL_ENVIRONMENTS = 50
NUM_TRAIN_ENVIRONMENTS = 40
MIN_TRIANGLE_AREA = 1.0


def set_global_seeds(seed: int) -> None:
    """Set seeds for reproducible experiment execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def triangle_area(a, b, c) -> float:
    """Compute 2D triangle area from three points."""
    return abs(
        0.5 * (
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
    )


def has_valid_beacon_geometry(beacon_positions: list, min_area: float = MIN_TRIANGLE_AREA) -> bool:
    """Return True when at least one beacon triplet forms a non-degenerate triangle."""
    if len(beacon_positions) < 3:
        return False
    for i, j, k in combinations(range(len(beacon_positions)), 3):
        if triangle_area(beacon_positions[i], beacon_positions[j], beacon_positions[k]) >= min_area:
            return True
    return False


# ============================================================
# STEP 1: Environment Generation
# ============================================================

def generate_random_environment(seed: int) -> dict:
    import numpy as np
    import random

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)  # ✅ Proper independent RNG

    mu, sigma = 20, 5
    min_grid, max_grid = 10, 30
    num_beacons = NUM_BEACONS

    # ---------- CLEAN NORMAL SAMPLING ----------
    def sample_truncated_normal():
        while True:
            val = np_rng.normal(mu, sigma)
            if min_grid <= val <= max_grid:
                return int(val)  # ❗ no round → smoother distribution

    grid_width = sample_truncated_normal()
    grid_height = sample_truncated_normal()

    # ---------- BEACON POSITIONS ----------
    beacon_positions = []
    min_distance = 2.0

    for _ in range(num_beacons):
        for _ in range(100):
            x = rng.uniform(1.0, grid_width - 1.0)
            y = rng.uniform(1.0, grid_height - 1.0)

            if all(
                np.hypot(x - bx, y - by) >= min_distance
                for bx, by in beacon_positions
            ):
                beacon_positions.append([round(x, 4), round(y, 4)])
                break
        else:
            beacon_positions.append([
                round(rng.uniform(1.0, grid_width - 1.0), 4),
                round(rng.uniform(1.0, grid_height - 1.0), 4),
            ])

    if not has_valid_beacon_geometry(beacon_positions):
        beacon_positions[-1][1] = round(
            min(grid_height - 1.0, beacon_positions[-1][1] + 1.0), 4
        )

    # ---------- LoS ----------
    los_probability = round(rng.uniform(0.4, 0.8), 4)

    # ---------- CIR ----------
    num_clusters = rng.randint(2, 4)
    rays_per_cluster = rng.randint(3, 5)
    los_max_clusters = rng.randint(2, min(4, num_clusters))
    delay_spread = round(rng.uniform(20, 100), 2)
    decay_factor = round(rng.uniform(0.9, 1.1), 4)

    # ---------- NOISE ----------
    los_std = round(rng.uniform(0.03, 0.08), 4)
    nlos_bias_min = round(rng.uniform(0.3, 0.5), 4)
    nlos_bias_max = round(rng.uniform(nlos_bias_min, 1.0), 4)

    # ---------- BATTERY ----------
    initial_battery = round(rng.uniform(80.0, 120.0), 2)
    consumption_multiplier = round(rng.uniform(2.0, 4.0), 4)

    return {
        'env_id': seed,
        'seed': seed,
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
        },
    }

def generate_all_environments(num_envs: int = TOTAL_ENVIRONMENTS) -> list:
    """Generate *num_envs* reproducible environment configs (seeds 0..N-1)."""
    configs = [generate_random_environment(seed=i) for i in range(num_envs)]
    
    # Validation: compute and print distribution statistics
    widths = [c['grid_width'] for c in configs]
    heights = [c['grid_height'] for c in configs]
    
    print(f"\n  Gaussian Distribution Validation (μ=20, σ=5, bounds=[10,30]):")
    print(f"    Widths  - mean: {np.mean(widths):.2f}, std: {np.std(widths):.2f}, range: [{min(widths)}, {max(widths)}]")
    print(f"    Heights - mean: {np.mean(heights):.2f}, std: {np.std(heights):.2f}, range: [{min(heights)}, {max(heights)}]")
    
    return configs


# ============================================================
# Distribution Visualization
# ============================================================

def plot_environment_distributions(configs: list, output_dir: Path) -> Path:
    """
    Plot and save histograms of grid dimensions with normal distribution overlay.
    
    Creates a 1x2 subplot figure showing the distribution of grid widths and heights
    with fitted normal distribution curves overlaid.
    """
    widths = np.array([c['grid_width'] for c in configs])
    heights = np.array([c['grid_height'] for c in configs])
    
    mu, sigma = 20, 5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot widths
    ax1.hist(widths, bins=12, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Sampled widths')
    x_range = np.linspace(widths.min() - 1, widths.max() + 1, 100)
    ax1.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu}, σ={sigma})')
    ax1.axvline(widths.mean(), color='green', linestyle='--', linewidth=2, label=f'Sample mean: {widths.mean():.2f}')
    ax1.set_xlabel('Grid Width', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Grid Width Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot heights
    ax2.hist(heights, bins=12, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label='Sampled heights')
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'b-', linewidth=2, label=f'Normal(μ={mu}, σ={sigma})')
    ax2.axvline(heights.mean(), color='green', linestyle='--', linewidth=2, label=f'Sample mean: {heights.mean():.2f}')
    ax2.set_xlabel('Grid Height', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Grid Height Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'environment_distributions.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"  Distribution plot saved → {plot_path}")
    plt.close()
    
    return plot_path


# ============================================================
# STEP 2: Train / Test Split
# ============================================================

def split_environments(configs, num_train: int = NUM_TRAIN_ENVIRONMENTS):
    """Split configs into train (first *num_train*) and test (rest)."""
    return configs[:num_train], configs[num_train:]


# ============================================================
# STEP 3a: DQN — one model per training environment
# ============================================================

def train_dqn_per_environment(
    train_configs: list,
    model_dir: Path,
    num_episodes: int = 500,
    max_steps: int = 2000,
):
    """
    Train a separate DQN for each training environment.

    Reuses `train_across_environments` from train_domain_generalization.py
    with a single-element config list per call so that every episode uses the
    same environment layout.
    """
    state_size = NUM_BEACONS

    for i, config in enumerate(train_configs):
        print(f"\n{'=' * 60}")
        print(f"DQN Per-Env  |  Training on environment {i}  "
              f"(grid {config['grid_width']}x{config['grid_height']})")
        print(f"{'=' * 60}")

        trainer = DQNTrainer(
            state_size=state_size,
            hidden_size=64,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=32,
            warmup_buffer_size=1000,
        )

        # Single-element list → always samples this environment
        train_across_environments(
            configs=[config],
            trainer=trainer,
            num_episodes=num_episodes,
            max_steps=max_steps,
        )

        model_path = model_dir / f'dqn_env_{i}.pt'
        trainer.save_model(str(model_path))


# ============================================================
# STEP 3b: Domain Generalization DQN — one model, all envs
# ============================================================

def train_domain_generalization_model(
    train_configs: list,
    model_dir: Path,
    num_episodes: int = 400,
    max_steps: int = 2000,
):
    """Train ONE domain-generalized DQN across all training environments."""
    state_size = NUM_BEACONS

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
        warmup_buffer_size=1000,
    )

    train_across_environments(
        configs=train_configs,
        trainer=trainer,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    model_path = model_dir / 'dqn_domain_gen.pt'
    trainer.save_model(str(model_path))


# ============================================================
# STEP 3c: Meta-RL (MAML-style)
# ============================================================

def train_meta_rl_model(
    train_configs: list,
    model_dir: Path,
    num_iterations: int = 100,
    tasks_per_batch: int = 4,
):
    """
    Train a Meta-RL model using existing `train_meta_rl()`.

    The existing script generates its own task configs internally via
    `generate_task_config()`.  We monkey-patch that function to sample
    from our training configs instead.
    """
    import rl.train_meta_rl as meta_rl_module

    # --- Monkey-patch task generation ---
    original_generate = meta_rl_module.generate_task_config

    def _patched_generate():
        cfg = random.choice(train_configs)
        return {
            'grid_size': max(cfg['grid_width'], cfg['grid_height']),
            'seed': int(cfg.get('seed', cfg['env_id'])),
        }

    meta_rl_module.generate_task_config = _patched_generate

    try:
        # Create model
        action_size = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        meta_model = MetaDQN(
            state_size=META_STATE_SIZE,
            action_size=action_size,
            hidden_size=64,
        ).to(device)

        # Delegate training to existing function
        trained_model = train_meta_rl(
            meta_model=meta_model,
            num_iterations=num_iterations,
            tasks_per_batch=tasks_per_batch,
            inner_steps=5,
            inner_lr=0.01,
            meta_lr=0.001,
            checkpoint_dir=str(model_dir.parent / 'checkpoints'),
        )

        model_path = model_dir / 'meta_rl.pt'
        torch.save(trained_model.state_dict(), str(model_path))
        print(f"Meta-RL model saved to {model_path}")
    finally:
        # Always restore original function
        meta_rl_module.generate_task_config = original_generate


# ============================================================
# STEP 3d: RL² LSTM
# ============================================================

def train_rl2_lstm_model(
    train_configs: list,
    model_dir: Path,
    num_episodes: int = 500,
):
    """
    Train an RL² LSTM model using existing `train_rl2()`.

    The existing script creates randomised environments internally via
    `create_randomized_environment()`.  We monkey-patch that function
    to build environments from our training configs.
    """
    import rl.train_rl2_lstm as rl2_module

    # --- Monkey-patch environment creation ---
    original_create = rl2_module.create_randomized_environment

    def _patched_create():
        cfg = random.choice(train_configs)
        grid_size = max(cfg['grid_width'], cfg['grid_height'])
        env = rl2_module.Environment(grid_size=grid_size)

        # Override beacon positions
        for idx, pos in enumerate(cfg['beacon_positions'][:len(env.beacons)]):
            env.beacons[idx].position = np.array(pos, dtype=float)

        # Randomise agent location
        rx = np.random.uniform(1.0, grid_size - 1.0)
        ry = np.random.uniform(1.0, grid_size - 1.0)
        env.agent.reset(x=rx, y=ry)

        # Randomise battery around config value
        for beacon in env.beacons:
            noise = np.random.uniform(0.8, 1.2)
            beacon.battery.battery = cfg['battery']['initial_level'] * noise

        return env

    rl2_module.create_randomized_environment = _patched_create

    try:
        model, log = train_rl2(
            num_episodes=num_episodes,
            num_train_steps=10,
            save_freq=50,
            eval_freq=50,
        )

        model_path = model_dir / 'rl2_lstm.pt'
        torch.save({'model_state_dict': model.state_dict()}, str(model_path))
        print(f"RL² LSTM model saved to {model_path}")
    finally:
        rl2_module.create_randomized_environment = original_create


# ============================================================
# Main entry point
# ============================================================

def main():
    set_global_seeds(42)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print(f"  UWB RL EXPERIMENT PIPELINE  —  {timestamp}")
    print("=" * 70)

    # Enable CIR-based measurements
    setup_cir_training(FAST_TRAINING)

    # Directories
    data_dir = ROOT / 'data'
    model_dir = ROOT / 'models'
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — Generate environments
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 1: GENERATING {TOTAL_ENVIRONMENTS} RANDOMIZED ENVIRONMENTS")
    print("=" * 70)

    configs_path = data_dir / 'env_configs.json'

    if configs_path.exists():
        print(f"Loading existing configs from {configs_path}")
        with open(configs_path, 'r') as f:
            all_configs = json.load(f)
        print(f"  Loaded {len(all_configs)} configs")
        if len(all_configs) != TOTAL_ENVIRONMENTS:
            print(
                f"  Config count mismatch (expected {TOTAL_ENVIRONMENTS}, "
                f"found {len(all_configs)}). Regenerating..."
            )
            all_configs = generate_all_environments(num_envs=TOTAL_ENVIRONMENTS)
            with open(configs_path, 'w') as f:
                json.dump(all_configs, f, indent=2)
            print(f"  Regenerated {TOTAL_ENVIRONMENTS} configs -> {configs_path}")
    else:
        all_configs = generate_all_environments(num_envs=TOTAL_ENVIRONMENTS)
        with open(configs_path, 'w') as f:
            json.dump(all_configs, f, indent=2)
        print(f"Generated {TOTAL_ENVIRONMENTS} configs -> {configs_path}")

    # Summary
    grid_sizes = [max(c['grid_width'], c['grid_height']) for c in all_configs]
    los_probs = [c['los_probability'] for c in all_configs]
    print(f"  Grid sizes : {min(grid_sizes)}-{max(grid_sizes)} "
          f"(avg {np.mean(grid_sizes):.0f})")
    print(f"  LoS probs  : {min(los_probs):.2f}-{max(los_probs):.2f} "
          f"(avg {np.mean(los_probs):.2f})")

    # Plot and save distribution
    print("\n" + "-" * 60)
    print("Visualizing Environment Distributions")
    print("-" * 60)
    plot_environment_distributions(all_configs, data_dir)

    # ------------------------------------------------------------------
    # STEP 2 — Train / Test split
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN / TEST SPLIT")
    print("=" * 70)

    train_configs, test_configs = split_environments(all_configs, num_train=NUM_TRAIN_ENVIRONMENTS)

    train_configs_path = data_dir / 'train_configs.json'
    test_configs_path = data_dir / 'test_configs.json'
    with open(train_configs_path, 'w') as f:
        json.dump(train_configs, f, indent=2)
    with open(test_configs_path, 'w') as f:
        json.dump(test_configs, f, indent=2)

    print(f"  Train : {len(train_configs)} environments  (env_ids {[c['env_id'] for c in train_configs]})")
    print(f"  Test  : {len(test_configs)} environments  (env_ids {[c['env_id'] for c in test_configs]})")
    print(f"  Saved train split -> {train_configs_path}")
    print(f"  Saved test split  -> {test_configs_path}")

    # ------------------------------------------------------------------
    # STEP 3 — Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING ALL MODELS")
    print("=" * 70)

    # 3a — DQN per environment
    print("\n" + "-" * 60)
    print(f"3a. DQN — Per Environment ({len(train_configs)} models)")
    print("-" * 60)
    train_dqn_per_environment(train_configs, model_dir)

    # 3b — Domain Generalization DQN (1 model)
    print("\n" + "-" * 60)
    print("3b. Domain Generalization DQN (1 model)")
    print("-" * 60)
    train_domain_generalization_model(train_configs, model_dir)

    # 3c — Meta-RL (1 model)
    print("\n" + "-" * 60)
    print("3c. Meta-RL (MAML) (1 model)")
    print("-" * 60)
    train_meta_rl_model(train_configs, model_dir)

    # 3d — RL² LSTM (1 model)
    print("\n" + "-" * 60)
    print("3d. RL² LSTM (1 model)")
    print("-" * 60)
    train_rl2_lstm_model(train_configs, model_dir)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Models saved in : {model_dir.resolve()}")
    print(f"  Env configs     : {configs_path.resolve()}")
    print(f"\n  Next step → python evaluate_generalization.py")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
