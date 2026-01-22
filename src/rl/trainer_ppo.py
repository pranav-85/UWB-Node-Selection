import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import Environment
from reward.reward import compute_reward
from config import NUM_BEACONS, NUM_SELECTED_BEACONS


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

class PPOTrainer:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=4,
        batch_size=64,
        device="cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.policy = PPOActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits, value = self.policy(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.buffer.states.append(state.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value.squeeze(0))

        return action.item()

def compute_returns_and_advantages(rewards, dones, values, gamma):
    returns = []
    advantages = []

    G = 0
    for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        if done:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    advantages = returns - values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

def ppo_update(trainer: PPOTrainer):
    # Convert numpy arrays to tensors
    states = torch.tensor(np.array(trainer.buffer.states), dtype=torch.float32).to(trainer.device)
    actions = torch.stack(trainer.buffer.actions).to(trainer.device)
    old_log_probs = torch.stack(trainer.buffer.log_probs).to(trainer.device)

    returns, advantages = compute_returns_and_advantages(
        trainer.buffer.rewards,
        trainer.buffer.dones,
        trainer.buffer.values,
        trainer.gamma
    )

    returns = returns.to(trainer.device)
    advantages = advantages.to(trainer.device)

    dataset_size = states.size(0)

    for _ in range(trainer.update_epochs):
        for idx in range(0, dataset_size, trainer.batch_size):
            slice_ = slice(idx, idx + trainer.batch_size)

            logits, values = trainer.policy(states[slice_])
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions[slice_])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[slice_])

            surr1 = ratio * advantages[slice_]
            surr2 = torch.clamp(
                ratio,
                1.0 - trainer.clip_eps,
                1.0 + trainer.clip_eps
            ) * advantages[slice_]

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns[slice_] - values.squeeze()).pow(2).mean()

            loss = (
                actor_loss
                + trainer.value_coef * critic_loss
                - trainer.entropy_coef * entropy
            )

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

    trainer.buffer.clear()

def train_ppo(
    env,
    trainer: PPOTrainer,
    num_episodes=200,
    max_steps=100
):
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            action = trainer.select_action(state)
            next_state, reward, done, _ = env.step(action)

            trainer.buffer.rewards.append(reward)
            trainer.buffer.dones.append(done)

            state = next_state
            ep_reward += reward

            if done:
                break

        ppo_update(trainer)
        episode_rewards.append(ep_reward)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1}, "
                f"Avg Reward (last 10): {sum(episode_rewards[-10:]) / 10:.3f}"
            )

    return episode_rewards


def main():
    """Train PPO agent for beacon selection with variable episode length.
    
    Per paper specification:
    - 500 training epochs
    - Variable episode length based on beacon battery depletion
    - Convergence observed around epoch 350
    """
    print("\n" + "="*70)
    print("TRAINING PPO AGENT FOR BEACON SELECTION (500 EPOCHS)")
    print("="*70 + "\n")
    
    # Generate all possible actions (beacon combinations)
    from itertools import combinations
    possible_actions = list(combinations(range(NUM_BEACONS), NUM_SELECTED_BEACONS))
    
    # Configuration
    state_dim = 2 + NUM_BEACONS + NUM_BEACONS  # agent position + batteries + los links
    action_dim = len(possible_actions)  # Number of possible beacon combinations
    num_episodes = 500  # Paper specifies 500 epochs
    batch_size = 64
    max_steps = 500  # Variable episode length with early termination
    
    # Initialize trainer
    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=batch_size
    )
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Possible beacon combinations: {action_dim}")
    print(f"Training for {num_episodes} episodes\n")
    
    # Visualization settings
    visualize = False  # Set to True to visualize every viz_freq episodes
    viz_freq = 50     # Visualize every N episodes
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    pbar = tqdm(range(num_episodes), desc="Training PPO", position=0)
    
    for episode in pbar:
        env = Environment()
        state = np.array(list(env.agent.get_position()) + env.get_battery_levels() + 
                        (env.current_links if env.current_links is not None else [0] * NUM_BEACONS),
                        dtype=np.float32)
        ep_reward = 0
        ep_length = 0
        
        for step in range(max_steps):
            env.step()
            
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, value = trainer.policy(state_tensor)
            
            # Create categorical distribution and sample action
            dist = Categorical(logits=logits[0])
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Ensure action is within valid range
            action_idx = action.item()
            if action_idx >= len(possible_actions):
                action_idx = action_idx % len(possible_actions)
            
            selected_beacons = list(possible_actions[action_idx])
            env.selected_beacon_indices = selected_beacons
            
            # Get reward
            agent_pos = np.array(env.agent.get_position())
            selected_positions = np.array([env.beacons[i].position for i in selected_beacons])
            los_flags = [env.current_links[i] for i in selected_beacons]
            battery_levels = env.get_battery_levels()
            reward = compute_reward(agent_pos, selected_positions, los_flags, battery_levels)
            
            # Store transition
            next_state = np.array(list(env.agent.get_position()) + env.get_battery_levels() + 
                                 (env.current_links if env.current_links is not None else [0] * NUM_BEACONS),
                                 dtype=np.float32)
            # Episode terminates if any beacon battery depleted or max_steps reached
            any_battery_depleted = any(level <= 0 for level in battery_levels)
            done = any_battery_depleted or step == max_steps - 1
            
            trainer.buffer.states.append(state)
            trainer.buffer.actions.append(action)
            trainer.buffer.log_probs.append(log_prob)
            trainer.buffer.rewards.append(reward)
            trainer.buffer.dones.append(done)
            trainer.buffer.values.append(value.item())
            
            state = next_state
            ep_reward += reward
            ep_length += 1
            
            if done:
                break
        
        # Update policy
        ppo_update(trainer)
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Visualize environment periodically
        if visualize and (episode + 1) % viz_freq == 0:
            import matplotlib.pyplot as plt
            fig, ax = env.visualize()
            fig.suptitle(f'Episode {episode + 1} - Reward: {ep_reward:.4f} - Length: {ep_length}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)
        
        # Update progress bar
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            pbar.set_postfix({
                'Avg Reward': f'{avg_reward:.4f}',
                'Avg Length': f'{avg_length:.0f}'
            })
    
    pbar.close()
    
    # Save model
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'ppo_model.pt'
    torch.save(trainer.policy.state_dict(), str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Total episodes: {num_episodes} (converged after ~350 per paper)")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.4f}")
    print(f"Max reward: {np.max(episode_rewards):.4f}")
    print(f"Min reward: {np.min(episode_rewards):.4f}")
    print(f"Avg episode length: {np.mean(episode_lengths[-10:]):.0f} steps (variable due to battery depletion)")
    print("="*70 + "\n")
    
    return episode_rewards


if __name__ == '__main__':
    main()
