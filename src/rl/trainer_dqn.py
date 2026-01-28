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


class DQN_MLP(nn.Module):
    """Deep Q Network with simple MLP (3 fully connected layers)."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize DQN with 3-layer MLP.
        
        Args:
            state_size: Size of input state
            action_size: Number of possible actions
            hidden_size: Number of hidden units in each layer
        """
        super(DQN_MLP, self).__init__()
        
        # 3 fully connected layers
        # FC1: state_size -> hidden_size
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # FC2: hidden_size -> hidden_size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # FC3: hidden_size -> action_size (output)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values of shape (batch_size, action_size)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Layer 3 (output)
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
                 epsilon_decay: float = 0.9999,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 warmup_buffer_size: int = 1000):
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
            warmup_buffer_size: Minimum buffer size before training starts (500-1000 recommended)
        """
        self.state_size = state_size
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
        self.q_network = DQN_MLP(state_size, self.action_size, hidden_size).to(self.device)
        self.target_network = DQN_MLP(state_size, self.action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def state_to_vector(self, env: Environment) -> np.ndarray:
        """
        Convert environment state to vector.
        
        State contains only observable system variables:
        - Battery levels of all beacons
        
        NOT included (unobservable in real system):
        - Ground-truth agent position (would provide unfair advantage)
        - LoS/NLoS flags (not directly observable)
        
        Args:
            env: Environment instance
        
        Returns:
            State vector containing only battery levels
        """
        battery_levels = env.get_battery_levels()
        
        # State = [battery_levels...] (NUM_BEACONS = 6 dimensions)
        state = np.array(battery_levels, dtype=np.float32)
        
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
            # Double DQN target
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
    
    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, 
              num_episodes: int = 500,
              max_steps: int = 2000,
              target_update_freq: int = 10,
              visualize: bool = False,
              viz_freq: int = 50,
              los_map_file: str = None):
        """
        Train the DQN agent with variable episode length based on battery depletion.
        
        Args:
            num_episodes: Number of training episodes (default 500 per paper)
            max_steps: Maximum steps per episode before termination
            target_update_freq: Frequency of target network updates
            visualize: Whether to visualize the environment during training
            viz_freq: Frequency of visualization (every N episodes)
            los_map_file: Path to pre-computed LoS map file. If None, generates new map.
        
        Note:
            Phase 1 - WARMUP: Collect transitions without training until buffer reaches warmup_buffer_size
            Phase 2 - TRAINING: Start training once buffer has enough experience
            Episode terminates early if any beacon battery is critically low (≤10%).
            This creates variable episode lengths based on selection policy quality.
        """
        episode_rewards = []
        episode_lengths = []
        
        # Create environment once - it persists across episodes
        # Only the agent location will change per episode
        env = Environment(los_map_file=los_map_file)
        
        # Outer progress bar for episodes
        pbar_episodes = tqdm(range(num_episodes), desc="Training Episodes", position=0)
        
        for episode in pbar_episodes:
            # Reset agent to random location for this episode
            env.reset_agent_to_random_location()
            # Reset all beacon batteries to 100% for this episode
            env.reset_beacon_batteries()
            state = self.state_to_vector(env)
            episode_reward = 0
            episode_length = 0
            losses = []
            
            # Inner progress bar for steps
            pbar_steps = tqdm(range(max_steps), desc=f"Episode {episode + 1} Steps", position=1, leave=False)
            
            for step in pbar_steps:
                # CORRECT FLOW: state → select_action → apply_action → next_state
                
                # 1. Select action based on current state
                action = self.select_action(state, training=True)
                selected_beacons = list(self.possible_actions[action])
                
                # 2. Apply beacon selection (part of the action)
                env.selected_beacon_indices = selected_beacons
                
                # 3. Step environment with the action applied
                env.step()
                
                # 4. Get next state and compute reward after transition
                agent_pos = env.agent.get_position()
                beacon_positions = [env.beacons[i].position for i in selected_beacons]
                los_flags = [env.current_links[i] for i in selected_beacons]
                battery_levels = env.get_battery_levels()
                
                # Reward depends on the action (beacon selection) and resulting state
                reward = compute_reward(agent_pos, beacon_positions, los_flags, battery_levels)
                
                # Reward clipping
                reward = np.clip(reward, -1.0, 1.0)
                
                # 5. Get next state AFTER transition
                next_state = self.state_to_vector(env)
                
                # 6. Check termination conditions
                # Critical threshold: 10% battery remaining (prevents degenerate zero-battery learning)
                min_battery = min(battery_levels)
                any_battery_critical = min_battery <= 10
                done = any_battery_critical or step == max_steps - 1
                
                # Store in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train only after buffer has enough transitions (warmup phase)
                # This stabilizes Q-learning by avoiding training on limited experience
                if len(self.replay_buffer) >= self.warmup_buffer_size:
                    loss = self.train_step()
                    if loss is not None:
                        losses.append(loss)
                else:
                    # During warmup, just collect transitions
                    loss = None
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Update progress bar with current stats
                if len(self.replay_buffer) < self.warmup_buffer_size:
                    # Show warmup progress
                    warmup_pct = (len(self.replay_buffer) / self.warmup_buffer_size) * 100
                    pbar_steps.set_postfix({
                        'Status': 'WARMUP',
                        'Buffer': f'{len(self.replay_buffer)}/{self.warmup_buffer_size}',
                        'Progress': f'{warmup_pct:.0f}%'
                    })
                else:
                    # Show training progress
                    pbar_steps.set_postfix({
                        'Reward': f'{reward:.4f}',
                        'Battery Min': f'{min(battery_levels):.2f}%',
                        'Avg Loss': f'{np.mean(losses[-10:]):.6f}' if losses else 'N/A'
                    })
                
                if done:
                    break
                # Decay epsilon PER STEP
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            pbar_steps.close()
            episode_lengths.append(episode_length)
            
            # Visualize environment periodically
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
            
            # Decay epsilon
            # self.decay_epsilon()
            
            episode_rewards.append(episode_reward)
            
            # Update main progress bar with episode stats
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


if __name__ == '__main__':
    # State size: only battery levels (observable) = 6 dimensions
    # NOT included: agent position (unobservable), LoS flags (unobservable)
    state_size = NUM_BEACONS
    
    # Create trainer
    trainer = DQNTrainer(
        state_size=state_size,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=32,
        warmup_buffer_size=1000  # Collect 1000 transitions before training
    )
    
    print(f"\n{'='*60}")
    print(f"Training Deep Q-Network with MLP")
    print(f"{'='*60}")
    print(f"Number of possible actions: {trainer.action_size}")
    print(f"State size: {state_size}")
    print(f"Warmup buffer size: {trainer.warmup_buffer_size}")
    print(f"{'='*60}\n")
    
    # Train (500 epochs as per paper, with variable episode length based on battery depletion)
    # Set visualize=True to see the environment during training every 50 episodes
    episode_rewards = trainer.train(num_episodes=500, max_steps=2000, target_update_freq=20, 
                                   visualize=False, los_map_file="src\los_maps\los_map_scenario_1.json")
    
    # Save model
    model_path = Path(__file__).parent.parent / 'models' / 'dqn_model.pt'
    model_path.parent.mkdir(exist_ok=True)
    trainer.save_model(str(model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total episodes: 500 (converged after ~350 per paper)")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")
