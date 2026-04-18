"""
Meta-RL Training using MAML-style Approach for UWB Node Selection.

This script implements Model-Agnostic Meta-Learning (MAML) for training a DQN agent
that can quickly adapt to new environments with only a few gradient steps.

Key Components:
- MetaDQN: Same architecture as standard DQN_MLP (3-layer MLP)
- inner_update(): Task-specific adaptation via few gradient steps
- meta_update(): Meta-policy update across task batch
- test_adaptation(): Evaluate improvement before/after adaptation

The meta-learning approach trains a "master" DQN that can be quickly fine-tuned
to any new environment with 5-10 adaptation steps.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import deque
import random
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
import copy

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.environment import Environment
from reward.reward import compute_reward
from localization.trilateration import uwb_trilateration_epoch

# Configuration
NUM_BEACONS = 6
NUM_SELECTED_BEACONS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaDQN(nn.Module):
    """
    Meta-DQN model for MAML-style meta-learning.
    Identical architecture to DQN_MLP but designed for parameter cloning and meta-updates.
    
    Architecture: state_size -> hidden_size (ReLU) -> hidden_size (ReLU) -> action_size
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize Meta-DQN.
        
        Args:
            state_size: Size of input state vector
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(MetaDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        """Forward pass through network."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Simple replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def generate_task_config():
    """
    Generate random environment configuration for a task.
    Each task represents a different environment with varying grid size.
    
    Note: num_beacons and los_probability are fixed from config.
    Environment variation is achieved through grid_size randomization.
    
    Returns:
        dict: Configuration with grid_size
    """
    grid_size = np.random.uniform(10, 30)  # 10-30m grid
    
    return {
        'grid_size': grid_size
    }


def create_environment_from_config(config: dict) -> Environment:
    """
    Create an Environment instance from a configuration dict.
    
    Args:
        config: Configuration dictionary with grid_size
    
    Returns:
        Environment: Configured environment instance
    """
    env = Environment(grid_size=config['grid_size'])
    return env


def state_to_vector(env: Environment) -> np.ndarray:
    """
    Convert environment state to state vector (battery levels only).
    
    Args:
        env: Environment instance
    
    Returns:
        np.ndarray: State vector of battery levels
    """
    battery_levels = env.get_battery_levels()
    return np.array(battery_levels, dtype=np.float32)


def select_action(model: nn.Module, state: np.ndarray, 
                  possible_actions: list, epsilon: float = 0.0) -> int:
    """
    Select action using epsilon-greedy policy.
    
    Args:
        model: DQN model
        state: Current state vector
        possible_actions: List of possible actions (beacon combinations)
        epsilon: Exploration rate (0 = greedy, 1 = random)
    
    Returns:
        int: Action index
    """
    if random.random() < epsilon:
        return random.randint(0, len(possible_actions) - 1)
    
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q_values = model(state_tensor)
        return q_values.argmax(dim=1).item()


def compute_dqn_loss(model: nn.Module, states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor, target_model: nn.Module = None,
                     gamma: float = 0.99) -> torch.Tensor:
    """
    Compute DQN loss for training.
    
    Args:
        model: Current Q-network
        states: Batch of states
        actions: Batch of action indices
        rewards: Batch of rewards
        next_states: Batch of next states
        dones: Batch of done flags
        target_model: Target network (if None, uses current model for targets)
        gamma: Discount factor
    
    Returns:
        torch.Tensor: Loss value
    """
    if target_model is None:
        target_model = model
    
    # Current Q-values
    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q-values
    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1)
        next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values
    
    # Loss
    loss = nn.SmoothL1Loss()(q_values, target_q_values)
    return loss


def inner_update(model: nn.Module, env: Environment, num_steps: int = 5,
                 inner_lr: float = 0.01, possible_actions: list = None,
                 epsilon: float = 0.1, gamma: float = 0.99) -> nn.Module:
    """
    Perform one inner loop update (task-specific adaptation).
    
    This function:
    1. Clones the input model parameters
    2. Collects transitions by rolling out in the environment
    3. Performs gradient-based optimization on the cloned model
    4. Returns the adapted model
    
    The original model is NOT modified.
    
    Args:
        model: Meta-model to adapt (not modified)
        env: Environment instance for this task
        num_steps: Number of adaptation steps
        inner_lr: Learning rate for inner loop
        possible_actions: List of possible actions
        epsilon: Exploration rate during collection
        gamma: Discount factor
    
    Returns:
        nn.Module: Adapted model with updated parameters
    """
    if possible_actions is None:
        possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    
    # Clone model to avoid modifying original
    adapted_model = copy.deepcopy(model)
    adapted_model.train()
    
    # Create optimizer for adapted model
    inner_optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)
    
    # Reset environment
    env.reset_agent_to_random_location()
    env.reset_beacon_batteries()
    state = state_to_vector(env)
    
    # Collect transitions and perform gradient steps
    replay_buffer = ReplayBuffer(capacity=1000)
    
    for step in range(num_steps):
        # Action selection
        action_idx = select_action(adapted_model, state, possible_actions, epsilon=epsilon)
        selected_beacons = list(possible_actions[action_idx])
        
        # Set selected beacons and step environment
        env.selected_beacon_indices = selected_beacons
        env.step()
        
        # Get environment state after step
        agent_pos = env.agent.get_position()
        beacon_positions = [env.beacons[i].position for i in selected_beacons]
        los_flags = [env.current_links[i] for i in selected_beacons]
        new_battery_levels = env.get_battery_levels()
        
        # Compute localization error
        try:
            result = uwb_trilateration_epoch(agent_pos, beacon_positions, los_flags)
            error = result['localization_error']
        except:
            error = 100.0  # Default error if trilateration fails
        
        # Compute reward
        reward = compute_reward(agent_pos, beacon_positions, los_flags, new_battery_levels)
        
        # Episode termination check
        done = any(battery <= 0.1 for battery in new_battery_levels)
        
        # Next state
        next_state = state_to_vector(env)
        
        # Store transition
        replay_buffer.push(state, action_idx, reward, next_state, done)
        
        state = next_state if not done else state_to_vector(env)
        if done:
            env.reset_agent_to_random_location()
            env.reset_beacon_batteries()
            state = state_to_vector(env)
    
    # Gradient step on collected transitions
    if len(replay_buffer) >= 4:
        for _ in range(2):  # 2 gradient steps per collection
            batch_size = min(4, len(replay_buffer))
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Compute loss
                loss = compute_dqn_loss(adapted_model, states, actions, rewards, 
                                       next_states, dones, target_model=adapted_model, gamma=gamma)
                
                # Gradient update
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
    
    return adapted_model


def meta_update(meta_model: nn.Module, task_batch: list, inner_lr: float = 0.01,
                meta_lr: float = 0.001, inner_steps: int = 5, gamma: float = 0.99) -> float:
    """
    Perform one meta-learning update (outer loop).
    
    This function:
    1. Processes each task through inner_update() to get adapted models
    2. Collects transitions using adapted models
    3. Computes meta-loss across adapted models
    4. Performs gradient update on meta_model
    
    Args:
        meta_model: Meta-model to update (modified in-place)
        task_batch: List of environment configurations (tasks)
        inner_lr: Learning rate for inner loop
        meta_lr: Learning rate for meta-update
        inner_steps: Number of adaptation steps
        gamma: Discount factor
    
    Returns:
        float: Meta-loss value
    """
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    
    meta_model.train()
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
    
    meta_loss_tensor = None
    num_tasks = len(task_batch)
    meta_loss_sum = 0.0
    
    for task_config in task_batch:
        # Create environment for this task
        env = create_environment_from_task_config(task_config)
        
        # Inner update: adapt to this task
        adapted_model = inner_update(meta_model, env, num_steps=inner_steps,
                                    inner_lr=inner_lr, possible_actions=possible_actions)
        
        # Rollout with adapted model on same task
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()
        state = state_to_vector(env)
        
        task_replay_buffer = ReplayBuffer(capacity=500)
        
        for step in range(10):  # 10 steps for meta-evaluation
            action_idx = select_action(adapted_model, state, possible_actions, epsilon=0.05)
            selected_beacons = list(possible_actions[action_idx])
            
            # Set selected beacons and step environment
            env.selected_beacon_indices = selected_beacons
            env.step()
            
            # Get environment state
            agent_pos = env.agent.get_position()
            beacon_positions = [env.beacons[i].position for i in selected_beacons]
            los_flags = [env.current_links[i] for i in selected_beacons]
            new_battery_levels = env.get_battery_levels()
            
            reward = compute_reward(agent_pos, beacon_positions, los_flags, new_battery_levels)
            
            done = any(battery <= 0.1 for battery in new_battery_levels)
            next_state = state_to_vector(env)
            
            task_replay_buffer.push(state, action_idx, reward, next_state, done)
            
            state = next_state if not done else state_to_vector(env)
            if done:
                env.reset_agent_to_random_location()
                env.reset_beacon_batteries()
                state = state_to_vector(env)
        
        # Compute meta-loss
        if len(task_replay_buffer) >= 4:
            batch_size = min(4, len(task_replay_buffer))
            states, actions, rewards, next_states, dones = task_replay_buffer.sample(batch_size)
            
            task_loss = compute_dqn_loss(meta_model, states, actions, rewards,
                                        next_states, dones, target_model=meta_model, 
                                        gamma=gamma)
            
            # Accumulate tensor loss
            if meta_loss_tensor is None:
                meta_loss_tensor = task_loss / num_tasks
            else:
                meta_loss_tensor = meta_loss_tensor + (task_loss / num_tasks)
            
            meta_loss_sum += task_loss.item()
    
    # Meta-update: backprop through meta_model
    if meta_loss_tensor is not None:
        meta_optimizer.zero_grad()
        meta_loss_tensor.backward()
        meta_optimizer.step()
    
    return meta_loss_sum / num_tasks if num_tasks > 0 else 0.0


def create_environment_from_task_config(config: dict) -> Environment:
    """
    Helper to create environment from task config.
    
    Args:
        config: Configuration dictionary with grid_size
    
    Returns:
        Environment: Configured environment
    """
    env = Environment(grid_size=config['grid_size'])
    return env


def train_meta_rl(meta_model: nn.Module, num_iterations: int = 100,
                  tasks_per_batch: int = 4, inner_steps: int = 5,
                  inner_lr: float = 0.01, meta_lr: float = 0.001,
                  checkpoint_dir: str = 'checkpoints') -> nn.Module:
    """
    Main meta-learning training loop.
    
    Args:
        meta_model: Meta-model to train
        num_iterations: Number of meta-training iterations
        tasks_per_batch: Number of tasks per meta-batch
        inner_steps: Number of inner loop adaptation steps
        inner_lr: Inner loop learning rate
        meta_lr: Meta-update learning rate
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        nn.Module: Trained meta-model
    """
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    
    os.makedirs(os.path.join(checkpoint_dir, 'meta_rl'), exist_ok=True)
    
    meta_losses = []
    
    print(f"Starting Meta-RL Training: {num_iterations} iterations, {tasks_per_batch} tasks/batch")
    print(f"Device: {DEVICE}")
    
    pbar = tqdm(range(num_iterations), desc="Meta-RL Iterations")
    
    for iteration in pbar:
        # Sample batch of tasks
        task_batch = [generate_task_config() for _ in range(tasks_per_batch)]
        
        # Meta-update
        meta_loss = meta_update(meta_model, task_batch, inner_lr=inner_lr,
                               meta_lr=meta_lr, inner_steps=inner_steps)
        
        meta_losses.append(meta_loss)
        
        # Logging
        pbar.set_postfix({'meta_loss': meta_loss})
        
        # Checkpoint every 10 iterations
        if (iteration + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, 'meta_rl',
                                          f'meta_dqn_{iteration+1}_{timestamp}.pt')
            torch.save(meta_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return meta_model


def test_adaptation(meta_model: nn.Module, num_test_tasks: int = 5,
                    inner_steps: int = 5, inner_lr: float = 0.01) -> dict:
    """
    Test meta-model adaptation on unseen tasks.
    
    For each test task:
    1. Evaluate meta_model WITHOUT adaptation (baseline)
    2. Perform few inner steps (adaptation)
    3. Evaluate adapted model
    4. Compute improvement metrics
    
    Args:
        meta_model: Trained meta-model
        num_test_tasks: Number of test tasks
        inner_steps: Number of adaptation steps
        inner_lr: Inner learning rate for adaptation
    
    Returns:
        dict: Results with error improvement and reward improvement
    """
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    meta_model.eval()
    
    error_improvements = []
    reward_improvements = []
    
    print(f"\nTesting adaptation on {num_test_tasks} unseen tasks...")
    
    for task_id in range(num_test_tasks):
        # Create new test task
        test_config = generate_task_config()
        env = create_environment_from_task_config(test_config)
        
        # --- Evaluate BEFORE adaptation ---
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()
        state = state_to_vector(env)
        
        errors_before = []
        rewards_before = []
        
        for step in range(20):  # 20 evaluation steps
            action_idx = select_action(meta_model, state, possible_actions, epsilon=0.0)
            selected_beacons = list(possible_actions[action_idx])
            
            # Set selected beacons and step
            env.selected_beacon_indices = selected_beacons
            env.step()
            
            agent_pos = env.agent.get_position()
            beacon_positions = [env.beacons[i].position for i in selected_beacons]
            los_flags = [env.current_links[i] for i in selected_beacons]
            new_battery_levels = env.get_battery_levels()
            
            # Compute localization error
            try:
                result = uwb_trilateration_epoch(agent_pos, beacon_positions, los_flags)
                error = result['localization_error']
            except:
                error = 100.0
            
            reward = compute_reward(agent_pos, beacon_positions, los_flags, new_battery_levels)
            
            errors_before.append(error)
            rewards_before.append(reward)
            
            done = any(battery <= 0.1 for battery in new_battery_levels)
            state = state_to_vector(env)
            if done:
                env.reset_agent_to_random_location()
                env.reset_beacon_batteries()
                state = state_to_vector(env)
        
        mean_error_before = np.mean(errors_before)
        mean_reward_before = np.mean(rewards_before)
        
        # --- Perform adaptation ---
        adapted_model = inner_update(meta_model, env, num_steps=inner_steps,
                                    inner_lr=inner_lr, possible_actions=possible_actions)
        adapted_model.eval()
        
        # --- Evaluate AFTER adaptation ---
        env.reset_agent_to_random_location()
        env.reset_beacon_batteries()
        state = state_to_vector(env)
        
        errors_after = []
        rewards_after = []
        
        for step in range(20):  # 20 evaluation steps
            action_idx = select_action(adapted_model, state, possible_actions, epsilon=0.0)
            selected_beacons = list(possible_actions[action_idx])
            
            # Set selected beacons and step
            env.selected_beacon_indices = selected_beacons
            env.step()
            
            agent_pos = env.agent.get_position()
            beacon_positions = [env.beacons[i].position for i in selected_beacons]
            los_flags = [env.current_links[i] for i in selected_beacons]
            new_battery_levels = env.get_battery_levels()
            
            # Compute localization error
            try:
                result = uwb_trilateration_epoch(agent_pos, beacon_positions, los_flags)
                error = result['localization_error']
            except:
                error = 100.0
            
            reward = compute_reward(agent_pos, beacon_positions, los_flags, new_battery_levels)
            
            errors_after.append(error)
            rewards_after.append(reward)
            
            done = any(battery <= 0.1 for battery in new_battery_levels)
            state = state_to_vector(env)
            if done:
                env.reset_agent_to_random_location()
                env.reset_beacon_batteries()
                state = state_to_vector(env)
        
        mean_error_after = np.mean(errors_after)
        mean_reward_after = np.mean(rewards_after)
        
        # Compute improvement
        error_improvement = ((mean_error_before - mean_error_after) / 
                            (mean_error_before + 1e-6)) * 100
        reward_improvement = ((mean_reward_after - mean_reward_before) / 
                             (abs(mean_reward_before) + 1e-6)) * 100
        
        error_improvements.append(error_improvement)
        reward_improvements.append(reward_improvement)
        
        print(f"  Task {task_id + 1}:")
        print(f"    Error: {mean_error_before:.4f}m -> {mean_error_after:.4f}m "
              f"({error_improvement:+.2f}% improvement)")
        print(f"    Reward: {mean_reward_before:.4f} -> {mean_reward_after:.4f} "
              f"({reward_improvement:+.2f}% improvement)")
    
    # Summary
    avg_error_improvement = np.mean(error_improvements)
    avg_reward_improvement = np.mean(reward_improvements)
    
    print(f"\nAdaptation Results Summary:")
    print(f"  Average Error Improvement: {avg_error_improvement:+.2f}%")
    print(f"  Average Reward Improvement: {avg_reward_improvement:+.2f}%")
    
    return {
        'error_improvements': error_improvements,
        'reward_improvements': reward_improvements,
        'avg_error_improvement': avg_error_improvement,
        'avg_reward_improvement': avg_reward_improvement
    }


def main():
    """Main entry point for meta-RL training."""
    print("=" * 80)
    print("UWB Node Selection - Meta-RL Training (MAML-style)")
    print("=" * 80)
    
    # Initialize meta-model
    state_size = NUM_BEACONS  # Battery levels only
    action_size = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))
    hidden_size = 64
    
    meta_model = MetaDQN(state_size, action_size, hidden_size).to(DEVICE)
    
    print(f"\nMeta-DQN Architecture:")
    print(f"  State Size: {state_size} (battery levels)")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Action Size: {action_size}")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in meta_model.parameters()):,}")
    
    # Training configuration
    num_iterations = 100
    tasks_per_batch = 4
    inner_steps = 5
    inner_lr = 0.01
    meta_lr = 0.001
    
    print(f"\nTraining Configuration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Tasks per Batch: {tasks_per_batch}")
    print(f"  Inner Loop Steps: {inner_steps}")
    print(f"  Inner Learning Rate: {inner_lr}")
    print(f"  Meta Learning Rate: {meta_lr}")
    
    # Train meta-model
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}\n")
    
    trained_meta_model = train_meta_rl(
        meta_model=meta_model,
        num_iterations=num_iterations,
        tasks_per_batch=tasks_per_batch,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        checkpoint_dir='checkpoints'
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'checkpoints/meta_rl/meta_dqn_final_{timestamp}.pt'
    os.makedirs('checkpoints/meta_rl', exist_ok=True)
    torch.save(trained_meta_model.state_dict(), checkpoint_path)
    print(f"\nFinal model saved: {checkpoint_path}")
    
    # Test adaptation on unseen tasks
    print(f"\n{'='*80}")
    print("Adaptation Testing on Unseen Tasks...")
    print(f"{'='*80}")
    
    adaptation_results = test_adaptation(
        meta_model=trained_meta_model,
        num_test_tasks=5,
        inner_steps=5,
        inner_lr=0.01
    )
    
    print(f"\n{'='*80}")
    print("Meta-RL Training Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
