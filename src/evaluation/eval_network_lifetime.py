import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from config import NUM_BEACONS, NUM_SELECTED_BEACONS

# ------------------------------------------------------------
# Beacon selection strategies
# ------------------------------------------------------------

def random_selection(env):
    return list(
        np.random.choice(NUM_BEACONS, NUM_SELECTED_BEACONS, replace=False)
    )


def nearest_neighbor_selection(env):
    agent_pos = np.array(env.agent.get_position())
    distances = []

    for i, beacon in enumerate(env.beacons):
        d = np.linalg.norm(agent_pos - np.array(beacon.position))
        distances.append((i, d))

    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:NUM_SELECTED_BEACONS]]


def dqn_selection(env, trainer):
    """RL-based beacon selection (DQN)."""
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


# ------------------------------------------------------------
# Network lifetime evaluation
# ------------------------------------------------------------

def measure_network_lifetime(
    method_name,
    selection_func,
    trainer=None,
    num_episodes=500,
    seed_offset=0,
    battery_threshold=10.0
):
    """
    Measure network lifetime (steps until battery failure).
    """
    lifetimes = []

    pbar = tqdm(range(num_episodes), desc=f"Lifetime: {method_name}")

    for episode in pbar:
        np.random.seed(seed_offset + episode)
        torch.manual_seed(seed_offset + episode)

        env = Environment()
        steps_alive = 0

        while True:
            # Select beacons
            if trainer is not None:
                selected_indices = selection_func(env, trainer)
            else:
                selected_indices = selection_func(env)

            env.selected_beacon_indices = selected_indices

            # Step environment
            env.step()
            steps_alive += 1

            # Check battery failure
            battery_levels = env.get_battery_levels()
            if min(battery_levels) <= battery_threshold:
                break

        lifetimes.append(steps_alive)

    lifetimes = np.array(lifetimes)

    return {
        "method": method_name,
        "mean": np.mean(lifetimes),
        "std": np.std(lifetimes),
        "min": np.min(lifetimes),
        "max": np.max(lifetimes),
        "all": lifetimes
    }


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    RESULTS_FILE = "results/network_lifetime_results.txt"

    results = []

    # -----------------------
    # Random
    # -----------------------
    results.append(
        measure_network_lifetime(
            method_name="Random",
            selection_func=random_selection
        )
    )

    # -----------------------
    # Nearest Neighbor
    # -----------------------
    results.append(
        measure_network_lifetime(
            method_name="Nearest Neighbor",
            selection_func=nearest_neighbor_selection
        )
    )

    # -----------------------
    # DQN
    # -----------------------
    from rl.trainer_dqn import DQNTrainer
    trainer = DQNTrainer(state_size=NUM_BEACONS)
    trainer.load_model("src/models/dqn_model.pt")

    results.append(
        measure_network_lifetime(
            method_name="DQN",
            selection_func=dqn_selection,
            trainer=trainer
        )
    )

    # --------------------------------------------------------
    # Save results to text file
    # --------------------------------------------------------
    
    output_path = Path(RESULTS_FILE)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w") as f:
        f.write("NETWORK LIFETIME EVALUATION\n")
        f.write("=" * 50 + "\n\n")

        for res in results:
            f.write(f"Method: {res['method']}\n")
            f.write(f"  Mean Lifetime (steps): {res['mean']:.2f}\n")
            f.write(f"  Std Lifetime:          {res['std']:.2f}\n")
            f.write(f"  Min Lifetime:          {res['min']}\n")
            f.write(f"  Max Lifetime:          {res['max']}\n")
            f.write("\n")

    print(f"\nNetwork lifetime results saved to '{RESULTS_FILE}'")


if __name__ == "__main__":
    main()
