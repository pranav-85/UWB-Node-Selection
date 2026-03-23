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


class PPO_Actor(nn.Module):
    """PPO Actor network - outputs action probabilities."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize Actor network with 3-layer MLP.
        
        Args:
            state_size: Size of input state
            action_size: Number of possible actions
            hidden_size: Number of hidden units in each layer
        """
        super(PPO_Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # Output: action logits
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            Action logits of shape (batch_size, action_size)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        logits = self.fc3(x)
        
        return logits


class PPO_Critic(nn.Module):
    """PPO Critic network - outputs state value."""
    
    def __init__(self, state_size: int, hidden_size: int = 64):
        """
        Initialize Critic network with 3-layer MLP.
        
        Args:
            state_size: Size of input state
            hidden_size: Number of hidden units in each layer
        """
        super(PPO_Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # Output: scalar value
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_size)
        
        Returns:
            State values of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        value = self.fc3(x)
        
        return value


class PPOTrajectoryBuffer:
    """Buffer for storing PPO trajectories."""
    
    def __init__(self):
        """Initialize trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to trajectory buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        """Get buffer contents and convert to tensors."""
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(np.array(self.actions), dtype=torch.long),
            torch.tensor(np.array(self.rewards), dtype=torch.float32),
            torch.tensor(np.array(self.values), dtype=torch.float32),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32),
            torch.tensor(np.array(self.dones), dtype=torch.float32)
        )
    
    def clear(self):
        """Clear all buffers."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(self, 
                 state_size: int,
                 hidden_size: int = 64,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 gradient_steps: int = 3):
        """
        Initialize PPO trainer.
        
        Args:
            state_size: Size of state vector
            hidden_size: Hidden layer size
            learning_rate: Learning rate for actor and critic
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            clip_ratio: Clipping ratio for PPO objective
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value function loss
            max_grad_norm: Maximum norm for gradient clipping
            gradient_steps: Number of gradient steps per update
        """
        self.state_size = state_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gradient_steps = gradient_steps
        
        # Generate all possible actions (combinations of 3 beacons from 6)
        self.possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
        self.action_size = len(self.possible_actions)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Actor and Critic networks
        self.actor = PPO_Actor(state_size, self.action_size, hidden_size).to(self.device)
        self.critic = PPO_Critic(state_size, hidden_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Trajectory buffer
        self.trajectory_buffer = PPOTrajectoryBuffer()
    
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
    
    def select_action(self, state: np.ndarray) -> tuple:
        """
        Select action using policy and get value estimate.
        
        Args:
            state: Current state vector
        
        Returns:
            Tuple of (action_index, log_prob, value_estimate)
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get action logits from actor
            logits = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Get value from critic
            value = self.critic(state_tensor)
            
            return action.item(), log_prob.item(), value.item()
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Trajectory rewards
            values: Trajectory values from critic
            dones: Done flags
            next_value: Value estimate at end of trajectory
        
        Returns:
            Advantages and returns
        """
        advantages = []
        returns = []
        
        gae = 0.0
        
        # Compute GAE backwards through trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = np.array(returns)
        
        return advantages, returns
    
    def train_step(self, advantages, returns):
        """
        Perform PPO training step on trajectory batch.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns
        """
        states, actions, rewards, values, log_probs, dones = self.trajectory_buffer.get()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = log_probs.to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Multiple gradient steps on same batch
        for _ in range(self.gradient_steps):
            
            # Actor loss
            logits = self.actor(states)
            action_dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy
            
            total_actor_loss = actor_loss + entropy_loss
            
            # Critic loss
            new_values = self.critic(states).squeeze(1)
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
            
            # Backward passes
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        return actor_loss.item(), value_loss.item(), entropy.item()
    
    def train(self, 
              num_episodes: int = 500,
              max_steps: int = 2000,
              visualize: bool = False,
              viz_freq: int = 50,
              los_map_file: str = None):
        """
        Train the PPO agent with variable episode length based on battery depletion.
        
        Args:
            num_episodes: Number of training episodes (default 500 per paper)
            max_steps: Maximum steps per episode before termination
            visualize: Whether to visualize the environment during training
            viz_freq: Frequency of visualization (every N episodes)
            los_map_file: Path to pre-computed LoS map file. If None, generates new map.
        
        Note:
            PPO collects trajectories and performs gradient updates at the end of each episode.
            Episode terminates early if any beacon battery is critically low (≤10%).
            This creates variable episode lengths based on selection policy quality.
        """
        episode_rewards = []
        episode_lengths = []
        
        # Create environment once - it persists across episodes
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
            
            # Clear trajectory buffer for new episode
            self.trajectory_buffer.clear()
            
            # Inner progress bar for steps
            pbar_steps = tqdm(range(max_steps), desc=f"Episode {episode + 1} Steps", position=1, leave=False)
            
            for step in pbar_steps:
                # CORRECT FLOW: state → select_action → apply_action → next_state
                
                # 1. Select action based on current state (with value estimate)
                action, log_prob, value = self.select_action(state)
                selected_beacons = list(self.possible_actions[action])
                
                # 2. Apply beacon selection
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
                min_battery = min(battery_levels)
                any_battery_critical = min_battery <= 10
                done = any_battery_critical or step == max_steps - 1
                
                # 7. Store in trajectory buffer
                self.trajectory_buffer.add(state, action, reward, value, log_prob, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Update progress bar with current stats
                pbar_steps.set_postfix({
                    'Reward': f'{reward:.4f}',
                    'Battery Min': f'{min(battery_levels):.2f}%',
                    'Buffer Size': len(self.trajectory_buffer)
                })
                
                if done:
                    break
            
            pbar_steps.close()
            episode_lengths.append(episode_length)
            
            # Compute final value estimate for advantage calculation
            if done:
                next_value = 0.0
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    next_value = self.critic(state_tensor).item()
            
            # Get trajectory data
            states, actions, rewards, values, log_probs, dones = self.trajectory_buffer.get()
            rewards_np = rewards.cpu().numpy()
            values_np = values.cpu().numpy()
            dones_np = dones.cpu().numpy()
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards_np, values_np, dones_np, next_value)
            
            # Perform PPO updates
            actor_loss, value_loss, entropy = self.train_step(advantages, returns)
            
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
            
            episode_rewards.append(episode_reward)
            
            # Update main progress bar with episode stats
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
            pbar_episodes.set_postfix({
                'Avg Reward': f'{avg_reward:.4f}',
                'Avg Length': f'{avg_length:.0f}',
                'Actor Loss': f'{actor_loss:.6f}',
                'Value Loss': f'{value_loss:.6f}',
                'Entropy': f'{entropy:.4f}'
            })
        
        pbar_episodes.close()
        return episode_rewards
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filepath)
        print(f"PPO Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"PPO Model loaded from {filepath}")


if __name__ == '__main__':
    # State size: only battery levels (observable) = 6 dimensions
    # NOT included: agent position (unobservable), LoS flags (unobservable)
    state_size = NUM_BEACONS
    
    # Create trainer
    trainer = PPOTrainer(
        state_size=state_size,
        hidden_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        gradient_steps=3
    )
    
    print(f"\n{'='*60}")
    print(f"Training Proximal Policy Optimization (PPO)")
    print(f"{'='*60}")
    print(f"Number of possible actions: {trainer.action_size}")
    print(f"State size: {state_size}")
    print(f"Clip ratio: {trainer.clip_ratio}")
    print(f"GAE lambda: {trainer.gae_lambda}")
    print(f"{'='*60}\n")
    
    # Train (500 episodes as per paper, with variable episode length based on battery depletion)
    episode_rewards = trainer.train(num_episodes=500, max_steps=2000, 
                                   visualize=False, los_map_file="src\los_maps\los_map_scenario_1.json")
    
    # Save model
    model_path = Path(__file__).parent.parent / 'checkpoints' / 'ppo_model.pt'
    model_path.parent.mkdir(exist_ok=True)
    trainer.save_model(str(model_path))
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total episodes: 500")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")
