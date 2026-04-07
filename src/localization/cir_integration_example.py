"""
Example: Integration of CIR-based distance measurements into existing trilateration pipeline.

This module demonstrates:
1. Using CIR model instead of simple noisy_distance()
2. Performance comparison between methods
3. Configuration options for RL training efficiency
"""

import numpy as np
from typing import List, Tuple

# Import both old and new models
from trilateration import trilateration_2d, localization_error, noisy_distance, compute_noisy_distances
from cir_model import (
    cir_based_distance_measurement,
    compute_cir_distances,
    CIRConfig,
    FAST_CIR_CONFIG,
    generate_cir,
    estimate_distance_from_cir
)


def uwb_trilateration_cir(
    target_pos: Tuple[float, float],
    beacon_positions: List[Tuple[float, float]],
    los_flags: List[bool],
    use_cir: bool = True,
    cir_config: CIRConfig = None
) -> Tuple[Tuple[float, float], List[float], float]:
    """
    Complete localization pipeline using CIR-based distance measurements.
    
    This is a drop-in replacement for uwb_trilateration_epoch() that uses
    the CIR model instead of simple noise.
    
    Args:
        target_pos: True target position (x, y)
        beacon_positions: List of beacon positions
        los_flags: List of LoS/NLoS flags
        use_cir: If True, use CIR model; if False, use original noisy model
        cir_config: CIRConfig for CIR generation (uses defaults if None)
    
    Returns:
        Tuple of (estimated_pos, distances, localization_error)
    """
    if use_cir:
        # Use CIR-based distances
        distances = compute_cir_distances(
            target_pos,
            beacon_positions,
            los_flags,
            config=cir_config
        )
    else:
        # Use original simple noisy model
        distances = compute_noisy_distances(
            target_pos,
            beacon_positions,
            los_flags
        )
    
    # Trilateration
    est_pos = trilateration_2d(beacon_positions, distances)
    
    # Localization error
    error = localization_error(target_pos, est_pos)
    
    return est_pos, distances, error


def compare_distance_estimation_methods():
    """
    Benchmark: Compare distance estimation methods across different scenarios.
    Useful for validation during development.
    """
    print("=" * 70)
    print("Distance Estimation Comparison: Simple Noise vs. CIR Model")
    print("=" * 70)
    
    # Test scenarios
    distances_to_test = [1.0, 3.0, 5.0, 8.0, 10.0]
    los_scenarios = [True, False]
    num_trials = 100
    
    for los in los_scenarios:
        print(f"\n{'LoS' if los else 'NLoS'} Scenario:")
        print("-" * 70)
        
        for true_distance in distances_to_test:
            # Simple noise method (original)
            simple_errors = []
            for _ in range(num_trials):
                d_noisy = noisy_distance(true_distance, los)
                error = abs(d_noisy - true_distance)
                simple_errors.append(error)
            
            # CIR method
            cir_errors = []
            for _ in range(num_trials):
                d_cir = cir_based_distance_measurement(
                    true_distance, los, 
                    method='first_path',
                    config=FAST_CIR_CONFIG
                )
                error = abs(d_cir - true_distance)
                cir_errors.append(error)
            
            simple_mean = np.mean(simple_errors)
            simple_std = np.std(simple_errors)
            cir_mean = np.mean(cir_errors)
            cir_std = np.std(cir_errors)
            
            print(f"  Distance {true_distance:.1f}m:")
            print(f"    Simple Noise: mean={simple_mean:.3f}m ± {simple_std:.3f}m")
            print(f"    CIR Model:    mean={cir_mean:.3f}m ± {cir_std:.3f}m")


def analyze_cir_characteristics():
    """
    Analyze CIR properties for LoS vs NLoS scenarios.
    Useful for understanding the behavior of the channel model.
    """
    print("\n" + "=" * 70)
    print("CIR Characteristics Analysis")
    print("=" * 70)
    
    distance = 5.0  # meters
    num_trials = 20
    
    for los in [True, False]:
        print(f"\n{'LoS' if los else 'NLoS'} at distance = {distance}m")
        print("-" * 70)
        
        delays_list = []
        powers_list = []
        toa_list = []
        
        for trial in range(num_trials):
            cir = generate_cir(distance, los, config=FAST_CIR_CONFIG)
            delays_list.append(cir['delays_s'])
            powers_list.append(cir['power_db'])
            
            est_dist, toa, pwr = estimate_distance_from_cir(cir, method='first_path')
            toa_list.append(toa)
        
        # Analyze multipath spread
        all_delays = np.concatenate(delays_list)
        all_powers = np.concatenate(powers_list)
        
        print(f"  Multipath components per channel:")
        print(f"    Mean: {len(all_delays) / num_trials:.1f}")
        print(f"    Min:  {min(len(d) for d in delays_list)}")
        print(f"    Max:  {max(len(d) for d in delays_list)}")
        
        if len(all_delays) > 0:
            delay_spread_ns = (np.max(all_delays) - np.min(all_delays)) * 1e9
            print(f"  Delay spread (across trials): {delay_spread_ns:.1f} ns")
        
        # ToA statistics
        toa_errors_m = [abs(t * 3e8 - distance) for t in toa_list]
        print(f"  ToA estimation error (first path method):")
        print(f"    Mean: {np.mean(toa_errors_m):.3f}m")
        print(f"    Std:  {np.std(toa_errors_m):.3f}m")
        print(f"    Max:  {np.max(toa_errors_m):.3f}m")


def example_localization_scenario():
    """
    Example: Single localization event comparing methods.
    """
    print("\n" + "=" * 70)
    print("Example Localization Scenario")
    print("=" * 70)
    
    # Setup
    target_pos = (5.0, 5.0)
    beacon_positions = [
        (1.0, 1.0),
        (9.0, 1.0),
        (1.0, 9.0)
    ]
    los_flags = [True, False, True]  # Mixed LoS/NLoS
    
    print(f"\nTarget position: {target_pos}")
    print(f"Beacons: {beacon_positions}")
    print(f"LoS flags: {los_flags}")
    
    # Method 1: Simple noise (original)
    print("\n--- Method 1: Simple Noise Model ---")
    est_pos_simple, dist_simple, error_simple = uwb_trilateration_cir(
        target_pos, beacon_positions, los_flags,
        use_cir=False
    )
    print(f"Estimated position: ({est_pos_simple[0]:.3f}, {est_pos_simple[1]:.3f})")
    print(f"Distances: {[f'{d:.3f}' for d in dist_simple]}")
    print(f"Localization error: {np.sqrt(error_simple):.3f}m")
    
    # Method 2: CIR model (fast config for RL)
    print("\n--- Method 2: CIR Model (Fast Config for RL) ---")
    est_pos_cir, dist_cir, error_cir = uwb_trilateration_cir(
        target_pos, beacon_positions, los_flags,
        use_cir=True,
        cir_config=FAST_CIR_CONFIG
    )
    print(f"Estimated position: ({est_pos_cir[0]:.3f}, {est_pos_cir[1]:.3f})")
    print(f"Distances: {[f'{d:.3f}' for d in dist_cir]}")
    print(f"Localization error: {np.sqrt(error_cir):.3f}m")


# Integration guide as comments
INTEGRATION_GUIDE = """
INTEGRATION GUIDE
=================

1. OPTION A: Use CIR model in your existing code (RECOMMENDED)
   
   In your trilateration.py or where you call compute_noisy_distances():
   
   OLD CODE:
   --------
   from trilateration import compute_noisy_distances
   distances = compute_noisy_distances(target_pos, beacon_positions, los_flags)
   
   NEW CODE:
   --------
   from cir_model import compute_cir_distances, FAST_CIR_CONFIG
   distances = compute_cir_distances(target_pos, beacon_positions, los_flags, 
                                     config=FAST_CIR_CONFIG)


2. OPTION B: Replace noisy_distance() function
   
   Create a new wrapper in trilateration.py:
   
   def noisy_distance_cir(d_true: float, los: bool) -> float:
       from cir_model import cir_based_distance_measurement, FAST_CIR_CONFIG
       return cir_based_distance_measurement(d_true, los, 
                                             config=FAST_CIR_CONFIG)
   
   Then replace: d_noisy = noisy_distance(...)
   With:         d_noisy = noisy_distance_cir(...)


3. CONFIGURATION FOR RL TRAINING
   
   Use FAST_CIR_CONFIG for faster simulation:
   - Reduced clusters (2 instead of 4)
   - Fewer rays per cluster (3 instead of 5)
   - Lower sample rate (1 GHz instead of 2 GHz)
   
   This reduces computation while maintaining realistic physics.


4. PERFORMANCE TUNING
   
   For even faster simulation, create custom config:
   
   from cir_model import CIRConfig
   
   ultra_fast_config = CIRConfig(
       max_clusters=1,
       max_rays_per_cluster=2,
       sample_rate=0.5e9,  # 500 MHz
       los_max_clusters=1
   )


5. BACKWARD COMPATIBILITY
   
   The original noisy_distance() function is unchanged, so you can:
   - Keep both models running in parallel for comparison
   - Gradually transition from one to the other
   - A/B test different configurations

"""

if __name__ == "__main__":
    # Run examples
    compare_distance_estimation_methods()
    analyze_cir_characteristics()
    example_localization_scenario()
    print(INTEGRATION_GUIDE)
