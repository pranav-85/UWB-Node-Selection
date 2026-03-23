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
from config import NUM_SELECTED_BEACONS, NUM_BEACONS


class DQN_LSTM(nn.Module):
    """Deep Q Network with LSTM (recurrent neural network)."""
    
    def __init__(self, state_size: int, action_size: int, lstm_hidden_size: int = 64, fc_hidden_size: int = 64, seq_length: int = 10):
        """
        Initialize DQN with LSTM layer.
        
        Args:
            state_size: Size of input state
            action_size: Number of possible actions
            lstm_hidden_size: Number of hidden units in LSTM layer
            fc_hidden_size: Number of hidden units in final fully connected layer
            seq_length: Length of input sequence for LSTM
        """
        super(DQN_LSTM, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_length = seq_length
        
        # LSTM layer: takes sequences of state vectors and produces a context vector
        self.lstm = nn.LSTM(input_size=state_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=1, 
                            batch_first=True)
        
        # Fully connected layers after LSTM
        # FC1: lstm_hidden_size -> fc_hidden_size
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.relu1 = nn.ReLU()
        
        # FC2: fc_hidden_size -> action_size (output)
        self.fc2 = nn.Linear(fc_hidden_size, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, state_size)
        
        Returns:
            Q-values of shape (batch_size, action_size)
        """
        # LSTM forward pass
        # Output: (batch_size, seq_length, lstm_hidden_size)
        # Hidden states not used, we use the final output
        lstm_output, (h_n, c_n) = self.lstm(x)
        
        # Take the last output from LSTM sequence
        # Shape: (batch_size, lstm_hidden_size)
        last_output = lstm_output[:, -1, :]
        
        # Apply fully connected layers
        x = self.fc1(last_output)
        x = self.relu1(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class SequenceReplayBuffer:
    """Experience replay buffer that maintains sequences for LSTM training."""
    
    def __init__(self, capacity: int = 10000, seq_length: int = 10):
        """Initialize replay buffer with sequence support."""
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        self.state_history = deque(maxlen=seq_length)
    
    def add(self, state, action, reward, next_state, done, next_state_history=None):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done, next_state_history))
    
    def update_state_history(self, state):
        """Update the running history of states."""
        self.state_history.append(state)
    
    def get_state_sequence(self):
        """Get padded state sequence for current history."""
        seq = list(self.state_history)
        
        # Pad with zeros if sequence is shorter than seq_length
        if len(seq) < self.seq_length:
            # Create zero vector - if seq is empty, use zero state for NUM_BEACONS
            if len(seq) == 0:
                zero_state = np.zeros(NUM_BEACONS, dtype=np.float32)
            else:
                zero_state = np.zeros_like(seq[0])
            padding = [zero_state] * (self.seq_length - len(seq))
            seq = padding + seq
        
        return np.array(seq)
    
    def sample(self, batch_size: int):
        """Sample a batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states_seq = []
        actions = []
        rewards = []
        next_states_seq = []
        dones = []
        
        for state, action, reward, next_state, done, next_state_history in batch:
            # Get state sequence (from history)
            state_seq = np.array(state)
            states_seq.append(state_seq)
            
            # Get next state sequence
            if next_state_history is not None:
                next_state_seq = np.array(next_state_history)
            else:
                next_state_seq = np.array(state)  # Fallback
            next_states_seq.append(next_state_seq)
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        return (
            torch.tensor(np.array(states_seq), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states_seq), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class LSTMTrainer:
    """Deep Q Learning trainer with LSTM."""
    
    def __init__(self, 
                 state_size: int,
                 lstm_hidden_size: int = 64,
                 fc_hidden_size: int = 64,
                 seq_length: int = 10,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9999,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 warmup_buffer_size: int = 1000):
        """
        Initialize LSTM DQN trainer.
        
        Args:
            state_size: Size of state vector
            lstm_hidden_size: Number of LSTM hidden units
            fc_hidden_size: Number of fully connected hidden units
            seq_length: Length of state sequences for LSTM
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            warmup_buffer_size: Minimum buffer size before training starts
        """
        self.state_size = state_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.warmup_buffer_size = warmup_buffer_size
        
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
        self.q_network = DQN_LSTM(state_size, self.action_size, lstm_hidden_size, fc_hidden_size, seq_length).to(self.device)
        self.target_network = DQN_LSTM(state_size, self.action_size, lstm_hidden_size, fc_hidden_size, seq_length).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = SequenceReplayBuffer(buffer_capacity, seq_length)
        
        # Training data storage (step-level details)
        self.training_data = []
    
    def state_to_vector(self, env: Environment) -> np.ndarray:
        """
        Convert environment state to vector.
        
        State contains only observable system variables:
        - Battery levels of all beacons
        
        Args:
            env: Environment instance
        
        Returns:
            State vector containing only battery levels
        """
        battery_levels = env.get_battery_levels()
        state = np.array(battery_levels, dtype=np.float32)
        return state
    
    def select_action(self, state_seq: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_seq: Current state sequence (seq_length, state_size)
            training: Whether in training mode
        
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            # Add batch dimension
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states_seq, actions, rewards, next_states_seq, dones = self.replay_buffer.sample(self.batch_size)
        states_seq = states_seq.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states_seq = next_states_seq.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states_seq)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states_seq).argmax(dim=1)
            next_q_values = self.target_network(next_states_seq) \
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
        Train the LSTM DQN agent.
        
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
        
        # Create environment
        env = Environment(los_map_file=los_map_file)
        
        # Initialize state history for LSTM
        self.replay_buffer.state_history.clear()
        initial_state = self.state_to_vector(env)
        for _ in range(self.seq_length):
            self.replay_buffer.update_state_history(initial_state)
        
        # Outer progress bar for episodes
        pbar_episodes = tqdm(range(num_episodes), desc="Training Episodes", position=0)
        
        for episode in pbar_episodes:
            # Reset environment for this episode
            env.reset_agent_to_random_location()
            env.reset_beacon_batteries()
            
            # Reset state history
            self.replay_buffer.state_history.clear()
            initial_state = self.state_to_vector(env)
            for _ in range(self.seq_length):
                self.replay_buffer.update_state_history(initial_state)
            
            state = initial_state
            episode_reward = 0
            episode_length = 0
            losses = []
            
            # Inner progress bar for steps
            pbar_steps = tqdm(range(max_steps), desc=f"Episode {episode + 1} Steps", position=1, leave=False)
            
            for step in pbar_steps:
                # Get current state sequence
                state_seq = self.replay_buffer.get_state_sequence()
                
                # 1. Select action based on current state sequence
                action = self.select_action(state_seq, training=True)
                selected_beacons = list(self.possible_actions[action])
                
                # Get predicted Q-values for all actions
                with torch.no_grad():
                    state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    predicted_value = q_values[0, action].item()
                    max_q_value = q_values[0].max().item()
                
                # 2. Apply beacon selection
                env.selected_beacon_indices = selected_beacons
                
                # 3. Step environment
                env.step()
                
                # 4. Get reward and observations
                agent_pos = env.agent.get_position()
                beacon_positions = [env.beacons[i].position for i in selected_beacons]
                los_flags = [env.current_links[i] for i in selected_beacons]
                battery_levels = env.get_battery_levels()
                
                reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
                reward = np.clip(reward, -1.0, 1.0)
                
                # 5. Get next state and update history
                next_state = self.state_to_vector(env)
                self.replay_buffer.update_state_history(next_state)
                next_state_seq = self.replay_buffer.get_state_sequence()
                
                # 6. Check termination
                min_battery = min(battery_levels)
                any_battery_critical = min_battery <= 10
                done = any_battery_critical or step == max_steps - 1
                
                # Store in replay buffer
                self.replay_buffer.add(state_seq, action, reward, next_state, done, next_state_seq)
                
                # Store detailed training data
                self.training_data.append({
                    'episode': episode + 1,
                    'step': step + 1,
                    'agent_x': agent_pos[0],
                    'agent_y': agent_pos[1],
                    'selected_beacons': str(selected_beacons),
                    'los_links': str(los_flags),
                    'battery_levels': str([f'{b:.2f}' for b in battery_levels]),
                    'predicted_q_value': predicted_value,
                    'max_q_value': max_q_value,
                    'reward': reward,
                    'epsilon': self.epsilon,
                    'buffer_size': len(self.replay_buffer)
                })
                
                # Train
                if len(self.replay_buffer) >= self.warmup_buffer_size:
                    loss = self.train_step()
                    if loss is not None:
                        losses.append(loss)
                        # Add loss to last training data point
                        self.training_data[-1]['loss'] = loss
                else:
                    loss = None
                    self.training_data[-1]['loss'] = None
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Update progress
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
            
            # Visualize
            if visualize and (episode + 1) % viz_freq == 0:
                import matplotlib.pyplot as plt
                fig, ax = env.visualize()
                fig.suptitle(f'Episode {episode + 1} - Reward: {episode_reward:.4f} - Length: {episode_length}', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(2)
                plt.close(fig)
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                self.update_target_network()
            
            episode_rewards.append(episode_reward)
            
            # Update main progress bar
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
        """Save detailed training data to CSV file (one row per step)."""
        if not self.training_data:
            print("No training data to save.")
            return
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = list(self.training_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data_point in self.training_data:
                writer.writerow(data_point)
        
        print(f"Training data saved to {filepath}")
        print(f"Total steps recorded: {len(self.training_data)}")


if __name__ == '__main__':
    # State size: battery levels (observable) = 6 dimensions
    state_size = NUM_BEACONS
    
    # Create trainer
    trainer = LSTMTrainer(
        state_size=state_size,
        lstm_hidden_size=64,
        fc_hidden_size=64,
        seq_length=10,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=32,
        warmup_buffer_size=1000
    )
    
    print(f"\n{'='*60}")
    print(f"Training Deep Q-Network with LSTM")
    print(f"{'='*60}")
    print(f"Number of possible actions: {trainer.action_size}")
    print(f"State size: {state_size}")
    print(f"Sequence length: {trainer.seq_length}")
    print(f"Warmup buffer size: {trainer.warmup_buffer_size}")
    print(f"{'='*60}\n")
    
    # Train
    episode_rewards = trainer.train(num_episodes=500, max_steps=2000, target_update_freq=20, 
                                   visualize=False, los_map_file="src\\los_maps\\los_map_scenario_1.json")
    
    # Save model
    model_path = Path(__file__).parent.parent / 'checkpoints' / 'lstm_model.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    
    # Save training data (step-level details)
    data_path = Path(__file__).parent.parent / 'data' / f'lstm_training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_training_data(str(data_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total episodes: 500")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Training data saved to: {data_path}")
    print(f"{'='*60}\n")
