"""
Saleh-Valenzuela Channel Impulse Response (CIR) Model for UWB Localization

Implements a simplified but realistic CIR model based on the Saleh-Valenzuela
multipath propagation model, suitable for efficient RL training simulations.

Key Features:
- Poisson processes for cluster and ray arrivals
- LoS vs NLoS channel characteristics
- Exponential amplitude decay with Nakagami fading
- ToA estimation from CIR peaks
- Distance conversion via speed of light
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class CIRConfig:
    """Configuration parameters for CIR generation."""
    # Sampling parameters
    sample_rate: float = 2e9          # 2 GHz (0.5 ns resolution)
    max_delay_ns: float = 200         # Max delay in nanoseconds (~60 m range)
    pulse_width_ns: float = 2.0       # Gaussian pulse width (FWHM)
    
    # LoS cluster/ray parameters
    los_cluster_rate: float = 0.06    # Clusters per ns
    los_ray_rate: float = 0.5         # Rays per ns within cluster
    los_decay_constant: float = 23.0  # Exponential decay time constant (ns)
    los_nakagami_m: float = 2.5       # Nakagami shape factor (LoS)
    
    # NLoS cluster/ray parameters
    nlos_cluster_rate: float = 0.1    # Higher for NLoS
    nlos_ray_rate: float = 1.2        # Higher for NLoS
    nlos_decay_constant: float = 40.0 # Slower decay for NLoS
    nlos_nakagami_m: float = 0.8      # Nakagami shape factor (NLoS)
    
    # LoS-specific adjustments
    los_direct_path_power: float = 1.0      # Power of direct path relative to first cluster
    los_max_clusters: int = 2                 # Limit clusters for LoS
    nlos_delay_shift: float = 15.0           # Delay bias for NLoS (ns)
    
    # Ray/cluster limits for efficiency
    max_clusters: int = 4             # Per channel
    max_rays_per_cluster: int = 5     # Per cluster
    
    # Speed of light
    c: float = 3e8                    # m/s


def generate_cir(
    distance_m: float,
    los: bool,
    config: CIRConfig = None,
    seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate a Channel Impulse Response (CIR) for a UWB link.
    
    Implements a simplified Saleh-Valenzuela model with:
    - Poisson-distributed cluster arrivals
    - Poisson-distributed rays within clusters
    - Exponential amplitude decay
    - Nakagami fading for amplitude variations
    
    Args:
        distance_m: True distance in meters
        los: Whether the link is Line-of-Sight
        config: CIRConfig object (uses defaults if None)
        seed: Optional random seed for reproducibility
    
    Returns:
        Dictionary containing:
        - 'delays_s': Array of ray delays (seconds)
        - 'amplitudes': Array of ray amplitudes (linear)
        - 'power_db': Array of ray powers (dB)
        - 'los_flag': Boolean indicating LoS
        - 'distance_m': True distance
    """
    if config is None:
        config = CIRConfig()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Direct path delay (speed of light propagation)
    direct_path_delay_ns = (distance_m / config.c) * 1e9
    
    # Select cluster/ray parameters based on LoS/NLoS
    if los:
        cluster_rate = config.los_cluster_rate
        ray_rate = config.los_ray_rate
        decay_constant = config.los_decay_constant
        nakagami_m = config.los_nakagami_m
        max_clusters = config.los_max_clusters
        delay_shift = 0
    else:
        cluster_rate = config.nlos_cluster_rate
        ray_rate = config.nlos_ray_rate
        decay_constant = config.nlos_decay_constant
        nakagami_m = config.nlos_nakagami_m
        max_clusters = config.max_clusters
        delay_shift = config.nlos_delay_shift
    
    delays_s = []
    amplitudes = []
    
    # Generate clusters using Poisson process
    num_clusters = np.random.poisson(max_clusters)
    num_clusters = min(num_clusters, max_clusters)
    
    if num_clusters == 0:
        num_clusters = 1  # At least one cluster
    
    # Cluster inter-arrival times (exponential Poisson process)
    cluster_delays_ns = np.cumsum(
        np.random.exponential(1.0 / cluster_rate, size=num_clusters)
    )
    
    # Add delay shift for NLoS
    cluster_delays_ns = cluster_delays_ns + delay_shift
    
    for cluster_idx, cluster_delay_ns in enumerate(cluster_delays_ns):
        # For LoS, add strong direct path at first cluster
        if los and cluster_idx == 0:
            direct_delay_s = direct_path_delay_ns / 1e9
            delays_s.append(direct_delay_s)
            # Direct path amplitude (normalized)
            amplitudes.append(config.los_direct_path_power)
        
        # Generate rays within this cluster
        num_rays = np.random.poisson(config.max_rays_per_cluster)
        num_rays = min(num_rays, config.max_rays_per_cluster)
        
        if num_rays == 0:
            num_rays = 1  # At least one ray per cluster
        
        # Ray inter-arrival times within cluster (exponential Poisson process)
        ray_delays_ns = np.cumsum(
            np.random.exponential(1.0 / ray_rate, size=num_rays)
        )
        
        for ray_delay_ns in ray_delays_ns:
            # Total delay: cluster delay + intra-cluster ray delay
            total_delay_ns = cluster_delay_ns + ray_delay_ns
            
            # Skip if beyond max delay window
            if total_delay_ns > config.max_delay_ns:
                continue
            
            # Exponential amplitude decay: alpha(t) = exp(-t / tau)
            base_amplitude = np.exp(-total_delay_ns / decay_constant)
            
            # Nakagami fading: multiply by random variable
            # Nakagami: draw from Nakagami distribution
            fading_amplitude = np.sqrt(
                np.random.gamma(nakagami_m, 1.0 / nakagami_m)
            )
            
            amplitude = base_amplitude * fading_amplitude
            
            # Convert delay to seconds
            delay_s = total_delay_ns / 1e9
            
            delays_s.append(delay_s)
            amplitudes.append(amplitude)
    
    # Convert to numpy arrays
    delays_s = np.array(delays_s)
    amplitudes = np.array(amplitudes)
    
    # Convert amplitudes to dB
    power_db = 10 * np.log10(np.maximum(amplitudes ** 2, 1e-10))
    
    return {
        'delays_s': delays_s,
        'amplitudes': amplitudes,
        'power_db': power_db,
        'los_flag': los,
        'distance_m': distance_m
    }


def simulate_received_signal(
    cir_data: Dict,
    config: CIRConfig = None,
    snr_db: float = 20
) -> Dict[str, np.ndarray]:
    """
    Simulate the received signal by convolving CIR with a pulse and adding noise.
    
    Args:
        cir_data: Output from generate_cir()
        config: CIRConfig object
        snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
        Dictionary containing:
        - 'signal': Received signal samples
        - 'time_s': Time axis (seconds)
        - 'power': Signal power (linear)
        - 'cir_data': Original CIR data
    """
    if config is None:
        config = CIRConfig()
    
    # Create time axis
    sample_period_s = 1.0 / config.sample_rate
    num_samples = int(config.max_delay_ns / 1e9 * config.sample_rate) + 1
    time_s = np.arange(num_samples) * sample_period_s
    
    # Create transmit pulse (Gaussian monocycle derivative)
    # Pulse is centered and delayed
    pulse_delay_s = 3 * config.pulse_width_ns / 1e9  # 3x width for tail
    pulse_sigma_s = (config.pulse_width_ns / 1e9) / (2 * np.sqrt(2 * np.log(2)))
    
    # Gaussian derivative (approximates UWB pulse spectrum)
    pulse = -(time_s - pulse_delay_s) * np.exp(
        -(time_s - pulse_delay_s)**2 / (2 * pulse_sigma_s**2)
    ) / (pulse_sigma_s ** 3)
    
    # Normalize pulse
    pulse = pulse / np.max(np.abs(pulse))
    
    # Create impulse response from CIR data
    cir_impulses = np.zeros(num_samples)
    delays = cir_data['delays_s']
    amplitudes = cir_data['amplitudes']
    
    for delay, amplitude in zip(delays, amplitudes):
        # Find closest sample
        sample_idx = int(np.round(delay / sample_period_s))
        if 0 <= sample_idx < num_samples:
            cir_impulses[sample_idx] += amplitude
    
    # Convolve pulse with CIR impulses
    received_signal = np.convolve(pulse, cir_impulses, mode='same')
    
    # Normalize received signal
    signal_power = np.mean(received_signal ** 2)
    if signal_power > 0:
        received_signal = received_signal / np.sqrt(signal_power)
        signal_power = 1.0
    
    # Add Gaussian noise based on SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=num_samples)
    received_signal = received_signal + noise
    
    return {
        'signal': received_signal,
        'time_s': time_s,
        'power': signal_power,
        'cir_data': cir_data,
        'noise_power': noise_power
    }


def estimate_distance_from_cir(
    cir_data: Dict,
    method: str = 'first_path',
    config: CIRConfig = None,
    power_threshold_db: float = -20
) -> Tuple[float, float, float]:
    """
    Estimate distance from CIR by extracting Time of Arrival (ToA).
    
    Two methods available:
    1. 'first_path': Uses the first detected path above threshold
    2. 'strongest_path': Uses the strongest path in the CIR
    
    Args:
        cir_data: Output from generate_cir()
        method: 'first_path' or 'strongest_path'
        config: CIRConfig object
        power_threshold_db: Minimum power threshold for path detection (dB)
    
    Returns:
        Tuple of (estimated_distance_m, toa_s, signal_power_db)
    """
    if config is None:
        config = CIRConfig()
    
    delays = cir_data['delays_s']
    power_db = cir_data['power_db']
    
    if len(delays) == 0:
        # Fallback: return a nominal distance
        return 0.1, 0, -40.0
    
    if method == 'first_path':
        # Find first path above threshold
        valid_indices = np.where(power_db >= power_threshold_db)[0]
        if len(valid_indices) > 0:
            toa_idx = valid_indices[0]
        else:
            # No path above threshold, use weakest path
            toa_idx = np.argmax(power_db)
    elif method == 'strongest_path':
        # Find strongest path
        toa_idx = np.argmax(power_db)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    toa_s = delays[toa_idx]
    signal_power_db = power_db[toa_idx]
    
    # Convert ToA to distance
    estimated_distance_m = toa_s * config.c
    
    return estimated_distance_m, toa_s, signal_power_db


def cir_based_distance_measurement(
    distance_m: float,
    los: bool,
    method: str = 'first_path',
    snr_db: float = 20,
    config: CIRConfig = None,
    add_measurement_noise: bool = True
) -> float:
    """
    High-level function: Generates CIR, simulates signal, and estimates distance.
    
    This function can be used as a drop-in replacement for the simple 
    noisy_distance() function in the existing trilateration pipeline.
    
    Args:
        distance_m: True distance in meters
        los: Whether link is LoS
        method: 'first_path' or 'strongest_path' for ToA estimation
        snr_db: Signal-to-Noise Ratio
        config: CIRConfig object
        add_measurement_noise: If True, adds random measurement uncertainty
    
    Returns:
        Estimated distance in meters
    """
    if config is None:
        config = CIRConfig()
    
    # Step 1: Generate CIR
    cir_data = generate_cir(distance_m, los, config)
    
    # Step 2: Simulate received signal (optional - can be skipped for speed)
    # For RL training efficiency, we can skip full signal simulation
    # and estimate directly from CIR
    
    # Step 3: Estimate distance from CIR
    est_distance_m, toa_s, signal_power_db = estimate_distance_from_cir(
        cir_data, method, config
    )
    
    # Step 4: Add optional measurement uncertainty
    if add_measurement_noise:
        # Small residual error from estimation process
        measurement_error = np.random.normal(0, 0.02)  # ~2cm std dev
        est_distance_m = np.maximum(est_distance_m + measurement_error, 0.01)
    
    return est_distance_m


def compute_cir_distances(
    target_pos: Tuple[float, float],
    beacon_positions: List[Tuple[float, float]],
    los_flags: List[bool],
    config: CIRConfig = None,
    method: str = 'first_path'
) -> List[float]:
    """
    Compute CIR-based distances for multiple beacons.
    
    This replaces compute_noisy_distances() while maintaining the same interface.
    
    Args:
        target_pos: (x_t, y_t) target position
        beacon_positions: [(x_i, y_i), ...] beacon positions
        los_flags: [True/False, ...] LoS flags
        config: CIRConfig object
        method: 'first_path' or 'strongest_path'
    
    Returns:
        List of estimated distances
    """
    if config is None:
        config = CIRConfig()
    
    x_t, y_t = target_pos
    distances = []
    
    for (x_b, y_b), los in zip(beacon_positions, los_flags):
        # Compute true distance
        d_true = np.sqrt((x_t - x_b)**2 + (y_t - y_b)**2)
        
        # Generate CIR-based estimated distance
        d_est = cir_based_distance_measurement(
            d_true, los, method=method, config=config
        )
        
        distances.append(d_est)
    
    return distances


# Convenience: Default configuration instances for quick use
DEFAULT_CIR_CONFIG = CIRConfig()
FAST_CIR_CONFIG = CIRConfig(
    max_clusters=2,
    max_rays_per_cluster=3,
    sample_rate=1e9  # 1 GHz for faster simulation
)
