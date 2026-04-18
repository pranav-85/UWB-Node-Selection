"""
RL² (Reinforcement Learning Squared) Training with LSTM-based agent.

This script implements RL² for UWB anchor selection where the LSTM hidden state
carries task-specific information instead of explicit parameter updates.

Key principles:
- Each episode = one randomized environment (task)
- LSTM hidden state persists within episode, resets between episodes
- Train on sequences of transitions
- No gradient updates during test adaptation (hidden state only)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, List, Optional
import sys
import random
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from localization.trilateration import trilateration_2d, compute_noisy_distances
from config import (
    GRID_SIZE, AGENT_INITIAL_X, AGENT_INITIAL_Y, AGENT_STEP_SIZE,
    NUM_BEACONS, BEACON_INITIAL_BATTERY, NUM_SELECTED_BEACONS,
    UWB_HARDWARE_PARAMS, LOS_PROBABILITY
)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1
STATE_SIZE = NUM_BEACONS  # Observable state: battery levels only [battery_0, ..., battery_5]
ACTION_SIZE = len(list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS)))  # C(6,3) = 20 beacon combinations
POSSIBLE_ACTIONS = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))  # List of beacon combinations
SEQUENCE_LENGTH = 20  # Length of sequences for training
BUFFER_SIZE = 1000  # Number of sequences in replay buffer
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000  # Update target network every N steps
MAX_EPISODE_LENGTH = 150


Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states', 'dones'))


class LSTM_DQN(nn.Module):
    """LSTM-based DQN for learning Q-values with task-dependent hidden state."""
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, num_layers: int = 1):
        """
        Args:
            input_size: Input state dimension
            hidden_size: LSTM hidden state dimension
            num_actions: Number of possible actions
            num_layers: Number of LSTM layers
        """
        super(LSTM_DQN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Q-value output layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
        self.relu = nn.ReLU()
    
    def forward(self, states: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through LSTM and Q-network.
        
        Args:
            states: Input states of shape (batch_size, seq_len, input_size) or (seq_len, input_size)
            hidden_state: Initial LSTM hidden state. If None, initialized to zeros.
        
        Returns:
            q_values: Q-values for each action (batch_size, seq_len, num_actions) or (seq_len, num_actions)
            next_hidden_state: LSTM hidden state after processing sequence
        """
        # Handle both batched and unbatched inputs
        is_batched = states.dim() == 3
        
        if not is_batched:
            states = states.unsqueeze(0)  # Add batch dimension
        
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=states.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=states.device)
            )
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(states, hidden_state)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Reshape for FC layers
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, self.hidden_size)
        
        # FC layers
        fc1_out = self.relu(self.fc1(lstm_out_flat))
        q_values = self.fc2(fc1_out)
        
        # Reshape back
        q_values = q_values.reshape(batch_size, seq_len, self.num_actions)
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            q_values = q_values.squeeze(0)
        
        return q_values, hidden_state


class SequenceReplayBuffer:
    """Replay buffer that stores and samples sequences of transitions."""
    
    def __init__(self, capacity: int, sequence_length: int):
        """
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of each sequence
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        """Store a sequence of transitions."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random sequences from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


def get_state(env: Environment) -> np.ndarray:
    """
    Construct state vector from environment observation.
    
    Observable State = [battery_0, ..., battery_5]
    Total: 6 dimensions (battery levels only - agent position and LoS are not observable)
    """
    battery_levels = np.array(env.get_battery_levels(), dtype=np.float32)
    return battery_levels


def create_randomized_environment() -> Environment:
    """
    Create a randomized environment for RL² task sampling.
    
    Randomization factors:
    - Agent position (random within grid)
    - Beacon battery levels (random initial values)
    - LoS probabilities can vary
    """
    env = Environment(grid_size=GRID_SIZE)
    
    # Randomize agent initial position
    random_x = np.random.uniform(1.0, GRID_SIZE - 1.0)
    random_y = np.random.uniform(1.0, GRID_SIZE - 1.0)
    env.agent.reset(x=random_x, y=random_y)
    
    # Randomize beacon battery levels (±20% of initial value)
    for beacon in env.beacons:
        noise = np.random.uniform(0.8, 1.2)
        beacon.battery.battery = BEACON_INITIAL_BATTERY * noise
    
    return env


def select_action(
    state: np.ndarray,
    model: LSTM_DQN,
    hidden_state: Optional[Tuple],
    epsilon: float,
    device: torch.device
) -> Tuple[int, Tuple]:
    """
    Select beacon combination using epsilon-greedy policy.
    
    Returns:
        action: Selected beacon combination index (0-19)
        hidden_state: Updated LSTM hidden state
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, ACTION_SIZE), hidden_state
    
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # (1, state_size)
        q_values, hidden_state = model(state_tensor, hidden_state)
        action = q_values.argmax(dim=-1).item()
    
    return action, hidden_state


def execute_action(env: Environment, action: int) -> None:
    """
    Execute beacon selection action in environment.
    
    Args:
        env: Environment instance
        action: Index into POSSIBLE_ACTIONS (0-19 for C(6,3) beacon combinations)
    """
    selected_beacons = POSSIBLE_ACTIONS[action]
    env.selected_beacon_indices = selected_beacons
    env.step()


def train_episode(
    env: Environment,
    model: LSTM_DQN,
    epsilon: float,
    device: torch.device
) -> Tuple[List, float]:
    """
    Run one episode and collect sequence of transitions.
    
    Returns:
        sequence: Transition sequence (states, actions, rewards, next_states, dones)
        episode_reward: Sum of rewards in episode
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Initialize hidden state for this episode
    hidden_state = None
    
    # Get initial state
    state = get_state(env)
    episode_reward = 0.0
    
    for step in range(MAX_EPISODE_LENGTH):
        states.append(state)
        
        # Select and execute action
        action, hidden_state = select_action(state, model, hidden_state, epsilon, device)
        actions.append(action)
        
        # Execute beacon selection
        execute_action(env, action)
        
        # Get next state
        next_state = get_state(env)
        next_states.append(next_state)
        
        # Compute reward based on localization quality
        agent_pos = np.array(env.agent.get_position())
        selected_beacons = POSSIBLE_ACTIONS[action]
        beacon_positions = np.array([env.beacons[i].position for i in selected_beacons])
        los_flags = [env.current_links[i] for i in selected_beacons]
        battery_levels = env.get_battery_levels()
        
        # Get noisy distances
        distances = compute_noisy_distances(agent_pos, beacon_positions, los_flags)
        
        # Estimate position via trilateration
        est_x, est_y = trilateration_2d(beacon_positions, distances)
        est_pos = np.array([est_x, est_y])
        
        # Localization error as primary reward signal
        localization_error = np.linalg.norm(agent_pos - est_pos)
        reward = -localization_error  # Negative error as reward (minimize error)
        
        # Add battery depletion penalty
        min_battery = min(battery_levels)
        if min_battery < 10.0:
            reward -= 10.0  # Penalize critical battery condition
        
        rewards.append(reward)
        episode_reward += reward
        
        # Check termination (critical battery)
        done = any(battery < 10.0 for battery in battery_levels)
        dones.append(done)
        
        state = next_state
        
        if done:
            break
    
    # Pad sequences to SEQUENCE_LENGTH
    current_len = len(states)
    if current_len < SEQUENCE_LENGTH:
        padding_size = SEQUENCE_LENGTH - current_len
        states.extend([states[-1]] * padding_size)
        actions.extend([actions[-1]] * padding_size)
        rewards.extend([0.0] * padding_size)
        next_states.extend([next_states[-1]] * padding_size)
        dones.extend([True] * padding_size)  # Mark padding as done
    else:
        # Truncate to SEQUENCE_LENGTH
        states = states[:SEQUENCE_LENGTH]
        actions = actions[:SEQUENCE_LENGTH]
        rewards = rewards[:SEQUENCE_LENGTH]
        next_states = next_states[:SEQUENCE_LENGTH]
        dones = dones[:SEQUENCE_LENGTH]
    
    # Convert to tensors
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(actions), dtype=torch.long)
    rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
    next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)
    
    sequence = Transition(
        states=states_tensor,
        actions=actions_tensor,
        rewards=rewards_tensor,
        next_states=next_states_tensor,
        dones=dones_tensor
    )
    
    return sequence, episode_reward


def compute_dqn_loss(
    model: LSTM_DQN,
    target_model: LSTM_DQN,
    sequence: Transition,
    gamma: float,
    device: torch.device
) -> torch.Tensor:
    """Compute DQN loss for a sequence of transitions."""
    states = sequence.states.to(device)  # (seq_len, state_size)
    actions = sequence.actions.to(device)  # (seq_len,)
    rewards = sequence.rewards.to(device)  # (seq_len,)
    next_states = sequence.next_states.to(device)  # (seq_len, state_size)
    dones = sequence.dones.to(device)  # (seq_len,)
    
    # Forward pass through model
    q_values, _ = model(states.unsqueeze(0), hidden_state=None)  # (1, seq_len, action_size)
    q_values = q_values.squeeze(0)  # (seq_len, action_size)
    
    # Select Q-values for taken actions
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (seq_len,)
    
    # Forward pass through target model
    with torch.no_grad():
        next_q_values, _ = target_model(next_states.unsqueeze(0), hidden_state=None)  # (1, seq_len, action_size)
        next_q_values = next_q_values.squeeze(0)  # (seq_len, action_size)
        next_q_max = next_q_values.max(dim=1)[0]  # (seq_len,)
    
    # Compute target
    target = rewards + gamma * next_q_max * (1 - dones)
    
    # MSE loss
    loss = nn.MSELoss()(q_selected, target)
    
    return loss


def train_rl2(
    num_episodes: int = 500,
    num_train_steps: int = 10,
    save_freq: int = 50,
    eval_freq: int = 50
):
    """
    Main RL² training loop.
    
    Args:
        num_episodes: Total episodes to train
        num_train_steps: Training steps per episode
        save_freq: Save checkpoint every N episodes
        eval_freq: Evaluate every N episodes
    """
    # Initialize model
    model = LSTM_DQN(
        input_size=STATE_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_actions=ACTION_SIZE,
        num_layers=LSTM_NUM_LAYERS
    ).to(DEVICE)
    
    target_model = LSTM_DQN(
        input_size=STATE_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_actions=ACTION_SIZE,
        num_layers=LSTM_NUM_LAYERS
    ).to(DEVICE)
    
    target_model.load_state_dict(model.state_dict())
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Replay buffer
    replay_buffer = SequenceReplayBuffer(capacity=BUFFER_SIZE, sequence_length=SEQUENCE_LENGTH)
    
    # Training log
    log = {
        'episodes': [],
        'episode_rewards': [],
        'train_losses': [],
        'timestamps': []
    }
    
    epsilon = EPSILON_START
    global_step = 0
    
    print(f"Starting RL² Training for Beacon Selection on {DEVICE}")
    print(f"Model: LSTM_DQN (hidden_size={LSTM_HIDDEN_SIZE})")
    print(f"State size: {STATE_SIZE} (battery levels)")  
    print(f"Action size: {ACTION_SIZE} (beacon combinations)")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print()
    
    for episode in range(num_episodes):
        # Create randomized environment for this task
        env = create_randomized_environment()
        
        # Collect sequence from environment
        sequence, episode_reward = train_episode(env, model, epsilon, DEVICE)
        replay_buffer.push(sequence)
        
        # Training step
        for train_step in range(num_train_steps):
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                
                total_loss = 0.0
                for seq in batch:
                    optimizer.zero_grad()
                    loss = compute_dqn_loss(model, target_model, seq, GAMMA, DEVICE)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    global_step += 1
                
                avg_loss = total_loss / len(batch)
                
                # Update target network
                if global_step % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Logging
        log['episodes'].append(episode)
        log['episode_rewards'].append(episode_reward)
        log['timestamps'].append(datetime.now().isoformat())
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(log['episode_rewards'][-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Epsilon: {epsilon:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'rl2_lstm'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_path / f'rl2_lstm_{episode + 1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'epsilon': epsilon,
                'log': log
            }, checkpoint_file)
            print(f"  [OK] Checkpoint saved: {checkpoint_file.name}")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_steps = evaluate_rl2(model, num_eval_episodes=5)
            print(f"  Evaluation: Avg Reward={eval_reward:.2f}, Avg Steps={eval_steps:.1f}")
    
    # Save final log
    log_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'rl2_lstm'
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\n[OK] Training complete. Log saved to {log_file}")
    
    return model, log


def evaluate_rl2(model: LSTM_DQN, num_eval_episodes: int = 5) -> Tuple[float, float]:
    """
    Evaluate RL² agent on randomized test environments.
    
    Measures agent's ability to adapt via hidden state (no gradient updates).
    
    Returns:
        avg_reward: Average reward across episodes
        avg_steps: Average steps taken
    """
    model.eval()
    
    total_reward = 0.0
    total_steps = 0
    
    for _ in range(num_eval_episodes):
        env = create_randomized_environment()
        
        # Initialize hidden state
        hidden_state = None
        state = get_state(env)
        episode_reward = 0.0
        steps = 0
        
        for step in range(MAX_EPISODE_LENGTH):
            # Select action (no exploration)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values, hidden_state = model(state_tensor, hidden_state)
                action = q_values.argmax(dim=-1).item()
            
            # Execute beacon selection
            execute_action(env, action)
            
            # Get next state
            next_state = get_state(env)
            
            # Compute reward based on localization quality
            agent_pos = np.array(env.agent.get_position())
            selected_beacons = POSSIBLE_ACTIONS[action]
            beacon_positions = np.array([env.beacons[i].position for i in selected_beacons])
            los_flags = [env.current_links[i] for i in selected_beacons]
            battery_levels = env.get_battery_levels()
            
            # Get noisy distances
            distances = compute_noisy_distances(agent_pos, beacon_positions, los_flags)
            
            # Estimate position via trilateration
            est_x, est_y = trilateration_2d(beacon_positions, distances)
            est_pos = np.array([est_x, est_y])
            
            # Localization error as primary reward signal
            localization_error = np.linalg.norm(agent_pos - est_pos)
            reward = -localization_error  # Negative error as reward
            
            # Add battery depletion penalty
            min_battery = min(battery_levels)
            if min_battery < 10.0:
                reward -= 10.0
            
            episode_reward += reward
            
            # Check termination
            done = any(battery < 10.0 for battery in battery_levels)
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        total_reward += episode_reward
        total_steps += steps
    
    model.train()
    
    return total_reward / num_eval_episodes, total_steps / num_eval_episodes


if __name__ == '__main__':
    # Train RL² agent
    model, log = train_rl2(
        num_episodes=500,
        num_train_steps=10,
        save_freq=50,
        eval_freq=50
    )
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {len(log['episodes'])}")
    print(f"Final reward: {log['episode_rewards'][-1]:.2f}")
    print(f"Best reward: {max(log['episode_rewards']):.2f}")
    print(f"Device: {DEVICE}")
