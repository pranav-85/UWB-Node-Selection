"""
Weighted Least Squares (WLS) based localization with Kalman filtering and NLoS bias compensation.

This module implements:
1. WLS estimator with link reliability weighting
2. NLoS bias modeling and compensation
3. Lightweight Kalman filter for temporal smoothing
"""

import numpy as np
from typing import Tuple, List
from scipy.optimize import least_squares
import warnings

# NLoS bias parameters
NLOS_BIAS_ESTIMATE = 0.5  # Estimated bias in meters for NLoS links


class KalmanFilter1D:
    """Simple 1D Kalman Filter for position smoothing."""
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        """
        Initialize Kalman filter.
        
        Args:
            process_variance: How much the state is expected to change (motion model)
            measurement_variance: Measurement noise variance
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.position = None
        self.velocity = None
        self.position_estimate = None
        self.position_error_estimate = 1.0
    
    def update(self, measured_position: float, dt: float = 1.0) -> float:
        """
        Update filter with new measurement.
        
        Args:
            measured_position: Measured position value
            dt: Time step (default 1.0)
        
        Returns:
            Filtered position estimate
        """
        if self.position is None:
            # First measurement
            self.position = measured_position
            self.velocity = 0.0
            self.position_estimate = measured_position
            return measured_position
        
        # Prediction step
        predicted_position = self.position + self.velocity * dt
        predicted_error = self.position_error_estimate + self.process_variance
        
        # Update step (Kalman gain)
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
        
        # State update
        self.position = predicted_position + kalman_gain * (measured_position - predicted_position)
        self.velocity = (self.position - self.position_estimate) / (dt + 1e-6)
        self.position_estimate = self.position
        self.position_error_estimate = (1 - kalman_gain) * predicted_error
        
        return self.position
    
    def reset(self):
        """Reset filter state."""
        self.position = None
        self.velocity = None
        self.position_estimate = None
        self.position_error_estimate = 1.0


class WLSLocalizer:
    """Weighted Least Squares localization with NLoS bias compensation."""
    
    def __init__(self, use_nlos_compensation: bool = True, use_kalman: bool = True):
        """
        Initialize WLS localizer.
        
        Args:
            use_nlos_compensation: Whether to apply NLoS bias compensation
            use_kalman: Whether to apply Kalman filtering
        """
        self.use_nlos_compensation = use_nlos_compensation
        self.use_kalman = use_kalman
        
        # Kalman filters for x and y positions
        self.kf_x = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)
        self.kf_y = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)
        
        self.last_estimate = None
    
    def estimate_distance_weights(self, distances: np.ndarray, los_flags: np.ndarray) -> np.ndarray:
        """
        Estimate reliability weights for each distance measurement.
        
        Weights are based on:
        1. Inverse distance (closer beacons more reliable)
        2. Link type (LoS more reliable than NLoS)
        
        Args:
            distances: Array of measured distances
            los_flags: Array of LoS/NLoS flags (True = LoS, False = NLoS)
        
        Returns:
            Array of weights (normalized to sum to number of beacons)
        """
        n = len(distances)
        weights = np.ones(n)
        
        # Weight based on distance (inverse) - closer is more reliable
        # Avoid division by very small distances
        safe_distances = np.maximum(distances, 0.1)
        distance_weights = 1.0 / safe_distances
        
        # Weight based on link type
        link_weights = np.ones(n)
        link_weights[~np.array(los_flags)] = 0.7  # NLoS links are less reliable
        
        # Combined weight
        weights = distance_weights * link_weights
        
        # Normalize
        weights = weights / np.sum(weights) * n
        
        return weights
    
    def compensate_nlos_bias(self, distances: np.ndarray, los_flags: np.ndarray) -> np.ndarray:
        """
        Compensate for NLoS bias in distance measurements.
        
        NLoS links typically have positive bias (measured distance > true distance).
        We reduce NLoS distances by estimated bias.
        
        Args:
            distances: Array of measured distances
            los_flags: Array of LoS/NLoS flags
        
        Returns:
            Bias-compensated distances
        """
        compensated = distances.copy()
        nlos_mask = ~np.array(los_flags)
        
        # Subtract estimated bias from NLoS distances
        # This is a simple compensation - in practice could be more sophisticated
        compensated[nlos_mask] = np.maximum(compensated[nlos_mask] - NLOS_BIAS_ESTIMATE, 0.1)
        
        return compensated
    
    def wls_solve(self, beacon_positions: np.ndarray, distances: np.ndarray, 
                  weights: np.ndarray, initial_guess: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Solve WLS problem using least squares optimization.
        
        Minimize: sum(w_i * (||x - b_i|| - d_i)^2)
        
        Args:
            beacon_positions: (N, 2) array of beacon positions
            distances: (N,) array of measured distances
            weights: (N,) array of measurement weights
            initial_guess: Initial estimate of agent position
        
        Returns:
            Tuple of (estimated_position, residual_error)
        """
        n_beacons = len(beacon_positions)
        
        # Initial guess (centroid if not provided)
        if initial_guess is None:
            x0 = np.mean(beacon_positions, axis=0)
        else:
            x0 = initial_guess
        
        def residuals(x):
            """Compute weighted residuals."""
            computed_distances = np.linalg.norm(beacon_positions - x, axis=1)
            residual = np.sqrt(weights) * (computed_distances - distances)
            return residual
        
        try:
            # Solve using least squares
            result = least_squares(residuals, x0, max_nfev=100, ftol=1e-6)
            estimated_pos = result.x
            
            # Compute final residual error
            final_residuals = residuals(estimated_pos)
            residual_error = np.sqrt(np.sum(final_residuals ** 2))
            
            return estimated_pos, residual_error
        except Exception as e:
            warnings.warn(f"WLS solve failed: {e}, returning initial guess")
            return x0, float('inf')
    
    def estimate(self, beacon_positions: np.ndarray, distances: np.ndarray, 
                 los_flags: np.ndarray, agent_estimate: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        Estimate agent position using WLS with NLoS compensation and Kalman filtering.
        
        Args:
            beacon_positions: (N, 2) array of beacon positions
            distances: (N,) array of measured distances
            los_flags: (N,) array of LoS/NLoS flags
            agent_estimate: Initial position estimate (used for weights)
        
        Returns:
            Tuple of (estimated_position, confidence_metric)
        """
        # Convert inputs
        beacon_positions = np.array(beacon_positions, dtype=np.float32)
        distances = np.array(distances, dtype=np.float32)
        los_flags = np.array(los_flags, dtype=bool)
        
        # Step 1: Compensate for NLoS bias
        if self.use_nlos_compensation:
            compensated_distances = self.compensate_nlos_bias(distances, los_flags)
        else:
            compensated_distances = distances
        
        # Step 2: Estimate reliability weights
        weights = self.estimate_distance_weights(distances, los_flags)
        
        # Step 3: Solve WLS problem
        if agent_estimate is None:
            agent_estimate = np.mean(beacon_positions, axis=0)
        
        wls_estimate, residual_error = self.wls_solve(beacon_positions, compensated_distances, 
                                                       weights, agent_estimate)
        
        # Step 4: Apply Kalman filtering
        if self.use_kalman:
            # Filter x and y positions separately
            filtered_x = self.kf_x.update(wls_estimate[0])
            filtered_y = self.kf_y.update(wls_estimate[1])
            final_estimate = np.array([filtered_x, filtered_y])
        else:
            final_estimate = wls_estimate
        
        # Confidence metric (inverse of residual)
        confidence = 1.0 / (1.0 + residual_error)
        
        self.last_estimate = final_estimate
        
        return final_estimate, confidence
    
    def reset(self):
        """Reset Kalman filters."""
        self.kf_x.reset()
        self.kf_y.reset()
        self.last_estimate = None


def compute_geometry_features(beacon_positions: np.ndarray, los_flags: np.ndarray, 
                               agent_estimate: np.ndarray) -> np.ndarray:
    """
    Compute geometry-related features for enhanced state representation.
    
    Features include:
    1. Estimated distances to each beacon
    2. Angle spread (angular separation) between beacons
    3. LoS ratio among selected beacons
    4. Centroid-to-beacon distances
    
    Args:
        beacon_positions: (N, 2) array of beacon positions
        los_flags: (N,) array of LoS/NLoS flags
        agent_estimate: (2,) array of estimated agent position
    
    Returns:
        Feature vector with geometry information
    """
    beacon_positions = np.array(beacon_positions, dtype=np.float32)
    agent_estimate = np.array(agent_estimate, dtype=np.float32)
    los_flags = np.array(los_flags, dtype=bool)
    
    features = []
    
    # 1. Estimated distances to selected beacons
    distances = np.linalg.norm(beacon_positions - agent_estimate, axis=1)
    features.extend(distances.tolist())
    
    # 2. Angle spread between selected beacons
    if len(beacon_positions) >= 2:
        # Compute angles from estimated position to each beacon
        angles = np.arctan2(beacon_positions[:, 1] - agent_estimate[1],
                           beacon_positions[:, 0] - agent_estimate[0])
        angle_spreads = []
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                angle_diff = np.abs(angles[i] - angles[j])
                # Normalize to [0, pi]
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                angle_spreads.append(angle_diff)
        
        if angle_spreads:
            features.append(np.mean(angle_spreads))
            features.append(np.std(angle_spreads) if len(angle_spreads) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])
    
    # 3. LoS ratio
    los_ratio = np.mean(los_flags)
    features.append(los_ratio)
    
    # 4. Distance statistics
    features.append(np.mean(distances))
    features.append(np.std(distances) if len(distances) > 1 else 0.0)
    
    return np.array(features, dtype=np.float32)


if __name__ == '__main__':
    # Test WLS localizer
    np.random.seed(42)
    
    # True agent position
    true_pos = np.array([5.0, 5.0])
    
    # Beacon positions
    beacons = np.array([
        [1.0, 1.0],
        [9.0, 9.0],
        [1.0, 9.0],
        [9.0, 1.0],
        [5.0, 1.0],
        [5.0, 9.0]
    ])
    
    # Simulate measurements
    true_distances = np.linalg.norm(beacons - true_pos, axis=1)
    
    # Add noise and NLoS bias
    measurement_noise = np.random.normal(0, 0.2, len(beacons))
    los_flags = np.array([True, False, True, False, True, True])
    
    distances = true_distances + measurement_noise
    distances[~los_flags] += 0.5  # Add NLoS bias
    
    # Estimate with WLS
    localizer = WLSLocalizer(use_nlos_compensation=True, use_kalman=True)
    estimate, confidence = localizer.estimate(beacons, distances, los_flags)
    
    print(f"True position: {true_pos}")
    print(f"WLS estimate: {estimate}")
    print(f"Error: {np.linalg.norm(true_pos - estimate):.4f} m")
    print(f"Confidence: {confidence:.4f}")
    
    # Test geometry features
    features = compute_geometry_features(beacons, los_flags, estimate)
    print(f"Geometry features: {features}")
