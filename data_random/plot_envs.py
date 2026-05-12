import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# Load your JSON file
with open("env_configs.json", "r") as f:
    data = json.load(f)

# Extract width and height
widths = [env["grid_width"] for env in data]
heights = [env["grid_height"] for env in data]

# ---------- SCATTER PLOT ----------
plt.figure(figsize=(6, 6))
plt.scatter(widths, heights)
plt.xlabel("Grid Width")
plt.ylabel("Grid Height")
plt.title("Environment Dimensions Scatter")
plt.grid()

plt.savefig("env_scatter.png", dpi=300, bbox_inches="tight")  # SAVE
plt.show()

# ---------- HISTOGRAM + NORMAL FIT (WIDTH) ----------
plt.figure(figsize=(6, 4))
plt.hist(widths, bins=10, density=True)

mu, std = np.mean(widths), np.std(widths)
x = np.linspace(min(widths), max(widths), 100)
plt.plot(x, norm.pdf(x, mu, std))

plt.title("Width Distribution with Normal Fit")
plt.xlabel("Width")
plt.ylabel("Density")

plt.savefig("width_distribution.png", dpi=300, bbox_inches="tight")  # SAVE
plt.show()

# ---------- HISTOGRAM + NORMAL FIT (HEIGHT) ----------
plt.figure(figsize=(6, 4))
plt.hist(heights, bins=10, density=True)

mu, std = np.mean(heights), np.std(heights)
x = np.linspace(min(heights), max(heights), 100)
plt.plot(x, norm.pdf(x, mu, std))

plt.title("Height Distribution with Normal Fit")
plt.xlabel("Height")
plt.ylabel("Density")

plt.savefig("height_distribution.png", dpi=300, bbox_inches="tight")  # SAVE
plt.show()