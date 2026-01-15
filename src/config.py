"""Configuration parameters for UWB Node Selection experiments."""

# Environment parameters
GRID_SIZE = 10
AGENT_INITIAL_X = GRID_SIZE / 2
AGENT_INITIAL_Y = GRID_SIZE / 2
AGENT_STEP_SIZE = 0.5

# Beacon parameters
NUM_BEACONS = 6
BEACON_INITIAL_BATTERY = 100.0

# Localization parameters
NUM_SELECTED_BEACONS = 3

# Simulation parameters
NUM_STEPS = 100
VISUALIZATION_FIGSIZE = (6, 6)
STEP_PAUSE_TIME = 1  # seconds

# Link model parameters
LOS_PROBABILITY = 0.5
