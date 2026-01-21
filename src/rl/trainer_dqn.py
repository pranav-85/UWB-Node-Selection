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

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from config import NUM_SELECTED_BEACONS, NUM_BEACONS


class DQN_CNN(nn.Module):
    """Deep Q Network with 1D CNN architecture."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize DQN with 1D CNN.
        
        Args:
            state_size: Size of input state
            action_size: Number of possible actions
            hidden_size: Number of hidden units
        """
        super(DQN_CNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Calculate flattened size after conv layers
        # State gets reshaped to (batch, 1, state_size)
        # After conv layers: (batch, 64, state_size)
        self.conv_output_size = 64 * state_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values of shape (batch_size, action_size)
        """
        # Reshape for conv1d: (batch_size, 1, state_size)
        x = x.unsqueeze(1)
        
        # Conv layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer."""
    
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


class DQNTrainer:
    """Deep Q Learning trainer with 1D CNN."""
    
    def __init__(self, 
                 state_size: int,
                 hidden_size: int = 64,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32):
        """
        Initialize DQN trainer.
        
        Args:
            state_size: Size of state vector
            hidden_size: Hidden layer size
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Generate all possible actions (combinations of 3 beacons from 6)
        self.possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
        self.action_size = len(self.possible_actions)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Networks
        self.q_network = DQN_CNN(state_size, self.action_size, hidden_size).to(self.device)
        self.target_network = DQN_CNN(state_size, self.action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def state_to_vector(self, env: Environment) -> np.ndarray:
        """
        Convert environment state to vector.
        
        Args:
            env: Environment instance
        
        Returns:
            State vector
        """
        agent_x, agent_y = env.agent.get_position()
        battery_levels = env.get_battery_levels()
        los_links = env.current_links if env.current_links is not None else [0] * NUM_BEACONS
        
        # Concatenate all features: [agent_x, agent_y, battery_levels..., los_links...]
        state = np.concatenate([
            [agent_x, agent_y],
            battery_levels,
            los_links
        ]).astype(np.float32)
        
        return state
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
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
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, 
              num_episodes: int = 100,
              max_steps: int = 100,
              target_update_freq: int = 10):
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            target_update_freq: Frequency of target network updates
        """
        episode_rewards = []
        
        # Outer progress bar for episodes
        pbar_episodes = tqdm(range(num_episodes), desc="Training Episodes", position=0)
        
        for episode in pbar_episodes:
            env = Environment()
            state = self.state_to_vector(env)
            episode_reward = 0
            losses = []
            
            # Inner progress bar for steps
            pbar_steps = tqdm(range(max_steps), desc=f"Episode {episode + 1} Steps", position=1, leave=False)
            
            for step in pbar_steps:
                # Select action
                action = self.select_action(state, training=True)
                selected_beacons = list(self.possible_actions[action])
                
                # Step environment
                env.step()
                env.selected_beacon_indices = selected_beacons
                
                # Get reward
                agent_pos = env.agent.get_position()
                beacon_positions = [env.beacons[i].position for i in selected_beacons]
                los_flags = [env.current_links[i] for i in selected_beacons]
                battery_levels = env.get_battery_levels()
                
                reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
                
                # Get next state
                next_state = self.state_to_vector(env)
                done = step == max_steps - 1
                
                # Store in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                # Update progress bar with current stats
                pbar_steps.set_postfix({
                    'Reward': f'{reward:.4f}',
                    'Avg Loss': f'{np.mean(losses[-10:]):.6f}' if losses else 'N/A'
                })
            
            pbar_steps.close()
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.decay_epsilon()
            
            episode_rewards.append(episode_reward)
            
            # Update main progress bar with episode stats
            avg_reward = np.mean(episode_rewards[-10:])
            pbar_episodes.set_postfix({
                'Avg Reward': f'{avg_reward:.4f}',
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


if __name__ == '__main__':
    # State size: agent_x, agent_y + 6 batteries + 6 los_links = 14
    state_size = 2 + NUM_BEACONS + NUM_BEACONS
    
    # Create trainer
    trainer = DQNTrainer(
        state_size=state_size,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=32
    )
    
    print(f"\n{'='*60}")
    print(f"Training Deep Q-Network with 1D CNN")
    print(f"{'='*60}")
    print(f"Number of possible actions: {trainer.action_size}")
    print(f"State size: {state_size}")
    print(f"{'='*60}\n")
    
    # Train
    episode_rewards = trainer.train(num_episodes=500, max_steps=150, target_update_freq=10)
    
    # Save model
    model_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
    model_path.parent.mkdir(exist_ok=True)
    trainer.save_model(str(model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total episodes: 100")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")
