"""
CIR Model Configuration for RL Training

This module provides utilities to enable/disable CIR-based distance measurements
in the RL training pipeline.

Usage:
    # At the start of training, call:
    from rl.cir_training_config import setup_cir_training, FAST_TRAINING, FULL_FIDELITY
    
    # For fast RL training (recommended):
    setup_cir_training(FAST_TRAINING)
    
    # For high-fidelity validation:
    setup_cir_training(FULL_FIDELITY)
    
    # Or disable CIR completely:
    setup_cir_training(SIMPLE_NOISE)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reward.reward import set_cir_mode, get_cir_mode
from localization.cir_model import CIRConfig, FAST_CIR_CONFIG, DEFAULT_CIR_CONFIG


# Configuration presets
SIMPLE_NOISE = {
    "name": "Simple Noise Model",
    "use_cir": False,
    "config": None,
    "description": "Original simple noise model - fastest but less realistic"
}

FAST_TRAINING = {
    "name": "CIR - Fast (RL Training)",
    "use_cir": True,
    "config": FAST_CIR_CONFIG,
    "description": "CIR with reduced fidelity for faster RL training"
}

FULL_FIDELITY = {
    "name": "CIR - Full Fidelity (Validation)",
    "use_cir": True,
    "config": DEFAULT_CIR_CONFIG,
    "description": "Full-fidelity CIR model for validation and analysis"
}

ULTRA_FAST = {
    "name": "CIR - Ultra Fast (Quick Testing)",
    "use_cir": True,
    "config": CIRConfig(
        max_clusters=1,
        max_rays_per_cluster=2,
        sample_rate=0.5e9,
        los_max_clusters=1
    ),
    "description": "Minimal CIR for rapid prototyping"
}

# Current active configuration
_active_config = SIMPLE_NOISE


def setup_cir_training(config_preset: dict = None, custom_cir_config: CIRConfig = None):
    """
    Setup CIR model for RL training.
    
    Args:
        config_preset: One of SIMPLE_NOISE, FAST_TRAINING, FULL_FIDELITY, ULTRA_FAST
                      If None, uses FAST_TRAINING as default
        custom_cir_config: Custom CIRConfig object to override preset
    
    Example:
        # Use fast training config
        setup_cir_training(FAST_TRAINING)
        
        # Use custom config
        custom_config = CIRConfig(max_clusters=3, max_rays_per_cluster=4)
        setup_cir_training(custom_cir_config=custom_config)
    """
    global _active_config
    
    if config_preset is None:
        config_preset = FAST_TRAINING
    
    # Override config if custom is provided
    if custom_cir_config is not None:
        config_preset = {
            "name": "Custom CIR Config",
            "use_cir": True,
            "config": custom_cir_config,
            "description": "User-defined CIR configuration"
        }
    
    _active_config = config_preset
    
    # Apply configuration to reward module
    set_cir_mode(
        use_cir=config_preset["use_cir"],
        cir_config=config_preset["config"]
    )
    
    print("\n" + "="*70)
    print(f"RL TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {config_preset['name']}")
    print(f"Description: {config_preset['description']}")
    if config_preset["use_cir"]:
        cfg = config_preset["config"]
        print(f"CIR Config:")
        print(f"  - Max clusters: {cfg.max_clusters}")
        print(f"  - Rays per cluster: {cfg.max_rays_per_cluster}")
        print(f"  - Sample rate: {cfg.sample_rate/1e9:.1f} GHz")
        print(f"  - LoS max clusters: {cfg.los_max_clusters}")
    print("="*70 + "\n")


def get_active_config() -> dict:
    """Get the currently active configuration."""
    return _active_config


def print_config_info():
    """Print information about available configurations."""
    print("\n" + "="*70)
    print("AVAILABLE CIR TRAINING CONFIGURATIONS")
    print("="*70)
    
    for config_name, config in [
        ("SIMPLE_NOISE", SIMPLE_NOISE),
        ("FAST_TRAINING", FAST_TRAINING),
        ("FULL_FIDELITY", FULL_FIDELITY),
        ("ULTRA_FAST", ULTRA_FAST)
    ]:
        print(f"\n{config_name}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        if config["use_cir"]:
            cfg = config["config"]
            print(f"  CIR Parameters:")
            print(f"    - Max clusters: {cfg.max_clusters}")
            print(f"    - Rays/cluster: {cfg.max_rays_per_cluster}")
            print(f"    - Sample rate: {cfg.sample_rate/1e9:.1f} GHz")
    print("\n" + "="*70)


def benchmark_configurations():
    """
    Quick benchmark comparing different configurations.
    Run this to see performance trade-offs.
    """
    import time
    import numpy as np
    from localization.cir_model import cir_based_distance_measurement
    
    print("\n" + "="*70)
    print("CIR TRAINING CONFIGURATIONS - PERFORMANCE BENCHMARK")
    print("="*70)
    
    distance = 5.0
    num_measurements = 100
    
    configs = [
        ("SIMPLE_NOISE (baseline)", SIMPLE_NOISE),
        ("ULTRA_FAST", ULTRA_FAST),
        ("FAST_TRAINING", FAST_TRAINING),
        ("FULL_FIDELITY", FULL_FIDELITY)
    ]
    
    results = []
    
    for config_name, config in configs:
        if config["use_cir"]:
            # Benchmark CIR
            start = time.time()
            for _ in range(num_measurements):
                d = cir_based_distance_measurement(
                    distance, True, config=config["config"]
                )
            elapsed = time.time() - start
            
            per_measurement = 1000 * elapsed / num_measurements  # milliseconds
            per_beacon = per_measurement  # 1 beacon per measurement in this test
            per_step_3beacons = per_measurement * 3  # 3 beacons typical
            per_episode = per_step_3beacons * 150  # 150 steps per episode
            
        else:
            # Benchmark simple noise (very fast)
            from localization.trilateration import noisy_distance
            start = time.time()
            for _ in range(num_measurements):
                d = noisy_distance(distance, True)
            elapsed = time.time() - start
            
            per_measurement = 1000 * elapsed / num_measurements
            per_beacon = per_measurement
            per_step_3beacons = per_measurement * 3
            per_episode = per_step_3beacons * 150
        
        results.append({
            "name": config_name,
            "per_meas_ms": per_measurement,
            "per_step_ms": per_step_3beacons,
            "per_episode_s": per_episode
        })
        
        print(f"\n{config_name}:")
        print(f"  Per measurement:     {per_measurement:6.2f} ms")
        print(f"  Per step (3 beacons): {per_step_3beacons:6.2f} ms")
        print(f"  Per episode (150 steps): {per_episode:6.2f} s")
    
    # Relative comparison
    print("\n" + "-"*70)
    print("RELATIVE SPEEDUP vs FULL_FIDELITY:")
    print("-"*70)
    full_fid_time = results[-1]["per_episode_s"]
    for result in results[:-1]:
        speedup = full_fid_time / result["per_episode_s"]
        print(f"  {result['name']:30s}: {speedup:5.1f}x faster")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_config_info()
    benchmark_configurations()
