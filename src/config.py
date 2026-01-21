"""Configuration parameters for UWB Node Selection experiments."""

# Environment parameters
GRID_SIZE = 10
AGENT_INITIAL_X = GRID_SIZE / 2
AGENT_INITIAL_Y = GRID_SIZE / 2
AGENT_STEP_SIZE = 0.5

# Beacon parameters
NUM_BEACONS = 4
BEACON_INITIAL_BATTERY = 100.0

# Beacon positions (relative to grid size)
BEACON_POSITIONS = [
    (0.5, 0.5),              # Bottom-left corner
    (GRID_SIZE - 0.5, 0.5),    # Bottom-right corner
    (0.5, GRID_SIZE - 0.5),    # Top-left corner
    (GRID_SIZE - 0.5, GRID_SIZE - 0.5),  # Top-right corner
]

# UWB Hardware Parameters
UWB_HARDWARE_PARAMS = {
    'P_COR': 10.0,
    'P_ADC': 15.0,
    'P_LNA': 9.0,
    'P_VGA': 5.0,
    'P_GEN': 8.0,
    'P_SYN': 6.0,
    'P_EST': 7.0,
    'T_SP': 0.0004,
    'T_PHR': 0.0002,
    'T_PAYLOAD': 0.002,
    'T_TR': 0.0001,
    'T_IPS': 0.00005,
    'T_ACK': 0.0002,
    'rho_c': 1,
    'rho_r': 1,
    'M': 1,
}

# Localization parameters
NUM_SELECTED_BEACONS = 3

# Simulation parameters
NUM_STEPS = 100
VISUALIZATION_FIGSIZE = (6, 6)
STEP_PAUSE_TIME = 1  # seconds

# Link model parameters
LOS_PROBABILITY = 0.5

# Reward function parameters
EPSILON = 1e-6  # Small constant to avoid division by zero
ER_TH = 2.0     # Localization error threshold
MD_TH = 0.5     # Mean deviation threshold