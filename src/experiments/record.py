import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.environment import Environment
from config import NUM_STEPS, VISUALIZATION_FIGSIZE, STEP_PAUSE_TIME

# Enable interactive mode
plt.ion()

env = Environment()

# Start recording
env.start_recording()

# Create ONE persistent figure
fig, ax = plt.subplots(figsize=VISUALIZATION_FIGSIZE)

for step in range(NUM_STEPS):
    env.step()

    # Clear previous drawing
    ax.clear()

    # Draw updated state on same axes
    env.visualize(title=f"Step {step + 1}", ax=ax)

    # Redraw canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(STEP_PAUSE_TIME)  # control animation speed

# Stop recording and save
env.stop_recording()
env.save_scenario(filename='test_scenario')

# Keep final frame visible
plt.ioff()
plt.show()
