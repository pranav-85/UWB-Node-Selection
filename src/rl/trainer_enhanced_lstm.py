"""
Enhanced LSTM-based DQN trainer with geometry-aware state representation and WLS-based position estimation.

This trainer improves upon the basic LSTM trainer by:
1. Using WLS with NLoS compensation and Kalman filtering for position estimation
2. Including geometry features in the state vector (distances, angle spreads, LoS ratio)
3. Better leveraging temporal information through LSTM
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from itertools import combinations
from tqdm import tqdm
import csv
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from localization.wls_kalman import WLSLocalizer, compute_geometry_features
from config import NUM_SELECTED_BEACONS, NUM_BEACONS


class Enhanced_DQN_MLP(nn.Module):
    """Enhanced DQN with MLP for improved state representation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize Enhanced DQN with MLP.
        
        Args:
            state_size: Size of input state (battery + geometry features)
            action_size: Number of possible actions
            hidden_size: Number of hidden units in each layer
        """
        super(Enhanced_DQN_MLP, self).__init__()
        
        # 4-layer MLP with geometry-aware features
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values of shape (batch_size, action_size)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        
        return x


class ReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class EnhancedDQNTrainer:
    """Enhanced DQN trainer with geometry-aware state and WLS-based localization."""
    
    def __init__(self, 
                 hidden_size: int = 128,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9999,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 warmup_buffer_size: int = 1000):
        """
        Initialize Enhanced DQN trainer with MLP and geometry-aware features.
        
        Args:
            hidden_size: Number of hidden units in MLP
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            warmup_buffer_size: Minimum buffer size before training starts
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.warmup_buffer_size = warmup_buffer_size
        
        # Generate all possible actions
        self.possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
        self.action_size = len(self.possible_actions)
        
        # Enhanced state size: 6 (battery) + 3 (distances) + 2 (angle features) + 1 (LoS ratio) + 2 (distance stats) = 14
        self.state_size = 6 + 3 + 2 + 1 + 2
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Networks
        self.q_network = Enhanced_DQN_MLP(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_network = Enhanced_DQN_MLP(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer (standard, not sequence-based)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # WLS localizer with Kalman filtering
        self.localizer = WLSLocalizer(use_nlos_compensation=True, use_kalman=True)
        
        # Training data storage
        self.training_data = []
    
    def build_enhanced_state(self, env: Environment, selected_beacons: list, 
                            agent_estimate: np.ndarray = None) -> np.ndarray:
        """
        Build enhanced state vector with battery levels and geometry features.
        
        Args:
            env: Environment instance
            selected_beacons: Indices of selected beacons
            agent_estimate: Current estimated agent position
        
        Returns:
            Enhanced state vector
        """
        battery_levels = np.array(env.get_battery_levels(), dtype=np.float32)
        
        # Get selected beacon positions and LoS flags
        beacon_positions = np.array([env.beacons[i].position for i in selected_beacons])
        los_flags = np.array([env.current_links[i] for i in selected_beacons])
        
        # Use centroid as initial estimate if not provided
        if agent_estimate is None:
            agent_estimate = np.mean(beacon_positions, axis=0)
        
        # Compute geometry features
        geometry_features = compute_geometry_features(beacon_positions, los_flags, agent_estimate)
        
        # Combine battery levels with geometry features
        state = np.concatenate([battery_levels, geometry_features])
        
        return state.astype(np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector (state_size,)
            training: Whether in training mode
        
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states) \
                                .gather(1, next_actions.unsqueeze(1)) \
                                .squeeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, 
              num_episodes: int = 500,
              max_steps: int = 2000,
              target_update_freq: int = 10,
              visualize: bool = False,
              viz_freq: int = 50,
              los_map_file: str = None):
        """
        Train the enhanced DQN agent with MLP and geometry-aware features.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            target_update_freq: Frequency of target network updates
            visualize: Whether to visualize
            viz_freq: Visualization frequency
            los_map_file: Path to LoS map file
        
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        episode_lengths = []
        
        env = Environment(los_map_file=los_map_file)
        
        pbar_episodes = tqdm(range(num_episodes), desc="Training Episodes", position=0)
        
        for episode in pbar_episodes:
            env.reset_agent_to_random_location()
            env.reset_beacon_batteries()
            self.localizer.reset()
            
            # Initialize estimated position
            estimated_pos = np.mean([env.beacons[i].position for i in range(3)], axis=0)
            # Build initial state
            state = self.build_enhanced_state(env, list(range(3)), estimated_pos)
            
            episode_reward = 0
            episode_length = 0
            losses = []
            
            pbar_steps = tqdm(range(max_steps), desc=f"Episode {episode + 1} Steps", position=1, leave=False)
            
            for step in pbar_steps:
                # Select action based on current state
                action = self.select_action(state, training=True)
                selected_beacons = list(self.possible_actions[action])
                
                # Get Q-values for the selected action
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    predicted_value = q_values[0, action].item()
                    max_q_value = q_values[0].max().item()
                
                # Apply beacon selection
                env.selected_beacon_indices = selected_beacons
                env.step()
                
                # Get observations
                agent_pos = env.agent.get_position()
                beacon_positions = np.array([env.beacons[i].position for i in selected_beacons])
                los_flags = np.array([env.current_links[i] for i in selected_beacons])
                battery_levels = env.get_battery_levels()
                
                # Estimate position using WLS with NLoS compensation and Kalman filtering
                estimated_pos, confidence = self.localizer.estimate(beacon_positions, 
                                                                     np.linalg.norm(beacon_positions - agent_pos, axis=1),
                                                                     los_flags,
                                                                     estimated_pos)
                
                reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
                reward = np.clip(reward, -1.0, 1.0)
                
                # Build next state with updated position estimate
                next_state = self.build_enhanced_state(env, selected_beacons, estimated_pos)
                
                # Check termination
                min_battery = min(battery_levels)
                done = min_battery <= 10 or step == max_steps - 1
                
                # Store in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Store training data
                self.training_data.append({
                    'episode': episode + 1,
                    'step': step + 1,
                    'agent_x': agent_pos[0],
                    'agent_y': agent_pos[1],
                    'estimated_x': estimated_pos[0],
                    'estimated_y': estimated_pos[1],
                    'selected_beacons': str(selected_beacons),
                    'los_links': str(los_flags.tolist()),
                    'battery_levels': str([f'{b:.2f}' for b in battery_levels]),
                    'predicted_q_value': predicted_value,
                    'max_q_value': max_q_value,
                    'confidence': confidence,
                    'reward': reward,
                    'epsilon': self.epsilon,
                })
                
                # Train
                if len(self.replay_buffer) >= self.warmup_buffer_size:
                    loss = self.train_step()
                    if loss is not None:
                        losses.append(loss)
                        self.training_data[-1]['loss'] = loss
                else:
                    self.training_data[-1]['loss'] = None
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if len(self.replay_buffer) < self.warmup_buffer_size:
                    warmup_pct = (len(self.replay_buffer) / self.warmup_buffer_size) * 100
                    pbar_steps.set_postfix({
                        'Status': 'WARMUP',
                        'Buffer': f'{len(self.replay_buffer)}/{self.warmup_buffer_size}',
                        'Progress': f'{warmup_pct:.0f}%'
                    })
                else:
                    pbar_steps.set_postfix({
                        'Reward': f'{reward:.4f}',
                        'Battery Min': f'{min(battery_levels):.2f}%',
                        'Avg Loss': f'{np.mean(losses[-10:]):.6f}' if losses else 'N/A'
                    })
                
                if done:
                    break
                
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            pbar_steps.close()
            episode_lengths.append(episode_length)
            
            if visualize and (episode + 1) % viz_freq == 0:
                import matplotlib.pyplot as plt
                fig, ax = env.visualize()
                fig.suptitle(f'Episode {episode + 1} - Reward: {episode_reward:.4f}', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(2)
                plt.close(fig)
            
            if (episode + 1) % target_update_freq == 0:
                self.update_target_network()
            
            episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
            pbar_episodes.set_postfix({
                'Avg Reward': f'{avg_reward:.4f}',
                'Avg Length': f'{avg_length:.0f}',
                'Epsilon': f'{self.epsilon:.4f}',
                'Buffer Size': len(self.replay_buffer)
            })
        
        pbar_episodes.close()
        return episode_rewards
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Model loaded from {filepath}")
    
    def save_training_data(self, filepath: str):
        """Save detailed training data to CSV file."""
        if not self.training_data:
            print("No training data to save.")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = list(self.training_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data_point in self.training_data:
                writer.writerow(data_point)
        
        print(f"Training data saved to {filepath}")
        print(f"Total steps recorded: {len(self.training_data)}")


if __name__ == '__main__':
    trainer = EnhancedDQNTrainer(
        hidden_size=128,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=32,
        warmup_buffer_size=1000
    )
    
    print(f"\n{'='*60}")
    print(f"Training Enhanced DQN (MLP) with Geometry-Aware State")
    print(f"{'='*60}")
    print(f"Number of possible actions: {trainer.action_size}")
    print(f"State size: {trainer.state_size}")
    print(f"Warmup buffer size: {trainer.warmup_buffer_size}")
    print(f"{'='*60}\n")
    
    episode_rewards = trainer.train(num_episodes=500, max_steps=2000, target_update_freq=20, 
                                   visualize=False, los_map_file="src\\los_maps\\los_map_scenario_1.json")
    
    model_path = Path(__file__).parent.parent / 'checkpoints' / 'enhanced_dqn_model.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    
    data_path = Path(__file__).parent.parent / 'data' / f'enhanced_dqn_training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_training_data(str(data_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Training data saved to: {data_path}")
    print(f"{'='*60}\n")
