import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from rl.trainer_dqn import DQNTrainer
from config import NUM_BEACONS, NUM_SELECTED_BEACONS


# ------------------------------------------------------------
# Selection Policies
# ------------------------------------------------------------

def random_selection(env):
    return list(np.random.choice(NUM_BEACONS, NUM_SELECTED_BEACONS, replace=False))


def nearest_neighbor_selection(env):
    agent_pos = np.array(env.agent.get_position())
    beacon_positions = np.array([beacon.position for beacon in env.beacons])
    distances = np.linalg.norm(beacon_positions - agent_pos, axis=1)
    return list(np.argsort(distances)[:NUM_SELECTED_BEACONS])


def gdop_selection(env):
    # Simple GDOP proxy using geometry spread
    agent_pos = np.array(env.agent.get_position())
    beacon_positions = np.array([beacon.position for beacon in env.beacons])

    best_score = float("inf")
    best_comb = None

    for comb in combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS):
        selected = beacon_positions[list(comb)]
        centroid = np.mean(selected, axis=0)
        spread = np.sum(np.linalg.norm(selected - centroid, axis=1))
        score = 1 / (spread + 1e-6)

        if score < best_score:
            best_score = score
            best_comb = comb

    return list(best_comb)


def dqn_selection(env, trainer):
    state = trainer.state_to_vector(env)
    action = trainer.select_action(state, training=False)
    return list(trainer.possible_actions[action])


# ------------------------------------------------------------
# Lifetime Evaluation
# ------------------------------------------------------------

def evaluate_lifetime(method_name,
                      selection_func,
                      trainer=None,
                      num_episodes=200,
                      max_steps=5000,
                      critical_threshold=10):

    lifetimes = []

    print(f"\nEvaluating {method_name}...")

    for episode in tqdm(range(num_episodes)):
        env = Environment()

        for step in range(max_steps):
            env.step()

            if trainer is not None:
                selected = selection_func(env, trainer)
            else:
                selected = selection_func(env)

            env.selected_beacon_indices = selected

            battery_levels = env.get_battery_levels()

            if min(battery_levels) <= critical_threshold:
                lifetimes.append(step)
                break
        else:
            # If never depleted
            lifetimes.append(max_steps)

    lifetimes = np.array(lifetimes)

    print(f"\nMethod: {method_name}")
    print(f"  Mean Lifetime (steps): {np.mean(lifetimes):.2f}")
    print(f"  Std Lifetime:          {np.std(lifetimes):.2f}")
    print(f"  Min Lifetime:          {np.min(lifetimes)}")
    print(f"  Max Lifetime:          {np.max(lifetimes)}")

    return lifetimes


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    # Load DQN model
    state_size = NUM_BEACONS
    trainer = DQNTrainer(state_size=state_size)

    model_path = Path(__file__).parent.parent / "models" / "dqn_model.pt"

    if model_path.exists():
        trainer.load_model(str(model_path))
        trainer.epsilon = 0.0
    else:
        print("Warning: DQN model not found!")

    print("\nNETWORK LIFETIME EVALUATION")
    print("="*50)

    random_life = evaluate_lifetime("Random", random_selection)
    gdop_life = evaluate_lifetime("GDOP", gdop_selection)
    nn_life = evaluate_lifetime("Nearest Neighbor", nearest_neighbor_selection)
    dqn_life = evaluate_lifetime("DQN", dqn_selection, trainer=trainer)