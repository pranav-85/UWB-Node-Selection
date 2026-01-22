"""Configuration parameters for UWB Node Selection experiments."""

# Environment parameters
GRID_SIZE = 10
AGENT_INITIAL_X = GRID_SIZE / 2
AGENT_INITIAL_Y = GRID_SIZE / 2
AGENT_STEP_SIZE = 0.5

# Beacon parameters
NUM_BEACONS = 6
BEACON_INITIAL_BATTERY = 100.0
BATTERY_CONSUMPTION_MULTIPLIER = 3.0  # Increase battery consumption per packet

# Beacon positions (relative to grid size)
BEACON_POSITIONS = [
    (0.5, 0.5),              # Bottom-left corner
    (5.0, 0.5),              # Bottom-middle
    (GRID_SIZE - 0.5, 0.5),  # Bottom-right corner
    (0.5, GRID_SIZE - 0.5),  # Top-left corner
    (5.0, GRID_SIZE - 0.5),  # Top-middle
    (GRID_SIZE - 0.5, GRID_SIZE - 0.5),  # Top-right corner
]

# UWB Hardware Parameters (from research paper)
UWB_HARDWARE_PARAMS = {
    'P_COR': 10.08,    # Correlator power (mW)
    'P_ADC': 2.2,      # ADC power (mW)
    'P_LNA': 9.4,      # Low Noise Amplifier power (mW)
    'P_VGA': 22.0,     # Variable Gain Amplifier power (mW)
    'P_GEN': 2.8,      # Generator power (mW)
    'P_SYN': 30.6,     # Synthesizer power (mW)
    'P_EST': 10.08,    # Estimator power (mW)
    'T_SP': 0.0004,    # Settling time (seconds)
    'T_PHR': 0.0002,   # PHR duration (seconds)
    'T_PAYLOAD': 0.002, # Payload duration (seconds)
    'T_TR': 0.0001,    # Transition time (seconds)
    'T_IPS': 0.00005,  # Inter-packet spacing (seconds)
    'T_ACK': 0.0002,   # ACK time (seconds)
    'rho_c': 1,        # Soft decision flag
    'rho_r': 1,        # Coherent receiver flag
    'M': 1,            # Number of correlator branches
}

# Localization parameters
NUM_SELECTED_BEACONS = 3

# Simulation parameters
NUM_STEPS = 100
VISUALIZATION_FIGSIZE = (6, 6)
STEP_PAUSE_TIME = 1  # seconds

# Link model parameters
LOS_PROBABILITY = 0.7

# Reward function parameters
EPSILON = 1e-6  # Small constant to avoid division by zero
ER_TH = 2.0     # Localization error threshold
MD_TH = 0.5     # Mean deviation threshold