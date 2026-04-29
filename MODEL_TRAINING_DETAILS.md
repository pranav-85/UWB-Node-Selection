# Model Training Details - UWB Beacon Selection System

**Project**: B.Tech Final Year Project - Optimal Selection of UWB Nodes in IoT Networks using Reinforcement Learning

**Date**: April 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Common Training Infrastructure](#common-training-infrastructure)
3. [Standard DQN](#standard-dqn)
4. [Domain Generalization DQN](#domain-generalization-dqn)
5. [Meta-RL with MAML](#meta-rl-with-maml)
6. [RL² with LSTM](#rl²-with-lstm)
7. [Training Environments](#training-environments)
8. [Comparison of Approaches](#comparison-of-approaches)

---

## Overview

This project implements and compares **four distinct reinforcement learning approaches** for intelligent UWB beacon selection:

| Model | Architecture | Strategy | Adaptation | Observations |
|-------|--------------|----------|-----------|--------------|
| **Standard DQN** | 3-layer MLP | Single env training | None | Battery levels only (6D) |
| **Domain Gen DQN** | 3-layer MLP | Multi-env training | None | Battery levels only (6D) |
| **Meta-RL (MAML)** | 3-layer MLP | Task-adaptive learning | Few gradient steps | Battery + Distance + LoS (15D) |
| **RL² LSTM** | LSTM + MLP | Recurrent learning | Hidden state only | Battery levels only (6D) |

**Problem Setup**:
- Select exactly **3 out of 6 beacons** for localization at each step
- **20 possible actions** ($\binom{6}{3}$)
- **Maximize**: Localization accuracy, battery efficiency, geometric quality
- **Minimize**: Power consumption, geometric dilution of precision (GDOP)

---

## Common Training Infrastructure

### Development Environment

**Software Stack**:
- **Python**: 3.13
- **Deep Learning Framework**: PyTorch 2.9.1 (CUDA 12.8)
- **Numerical Computing**: NumPy 2.4.1
- **Visualization**: Matplotlib 3.10.8
- **Device**: NVIDIA RTX 3050 (8GB VRAM) with CUDA 12.8 support

**Hardware Capabilities**:
```
GPU: NVIDIA RTX 3050
CUDA Version: 12.8
Device Memory: 8GB
Automatic CPU Fallback: Enabled
```

### Common Hyperparameters

#### Network Architecture (DQN, Domain Gen, Meta-RL)

All three classical approaches use an identical **3-layer MLP**:

```
Input Layer:        state_size → 64 neurons (ReLU)
Hidden Layer 1:     64 → 64 neurons (ReLU)
Output Layer:       64 → 20 neurons (Q-values)
```

**Architecture Details**:
- **Layer 1** (state_size → 64): `nn.Linear(state_size, 64) + ReLU`
- **Layer 2** (64 → 64): `nn.Linear(64, 64) + ReLU`
- **Layer 3** (64 → 20): `nn.Linear(64, 20)` (no activation - raw Q-values)
- **Total Parameters**: ~2,500 (state_size-dependent)

#### Optimization Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | Adam | Adaptive moment estimation |
| **Learning Rate** | 0.001 | Initial learning rate for gradient descent |
| **Weight Decay (L2)** | 1e-5 | L2 regularization to prevent overfitting |
| **Gradient Clipping** | None | Unbounded gradients |
| **Batch Normalization** | None | No normalization layers |

#### DQN Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Discount Factor (γ)** | 0.99 | Future reward decay |
| **Epsilon Start** | 1.0 | 100% random exploration initially |
| **Epsilon End** | 0.01-0.05 | Minimum exploration rate |
| **Epsilon Decay** | 0.9999-0.995 | Decay per episode |
| **Replay Buffer Capacity** | 10,000 | Maximum stored transitions |
| **Batch Size** | 32 | Gradient update batch size |
| **Warmup Buffer Size** | 1,000 | Min transitions before training starts |
| **Target Update Frequency** | 1,000 steps | Sync target network every N steps |
| **Loss Function** | Smooth L1 Loss | Huber loss for stability |

**Double DQN Implementation**:
- Use **current Q-network** to select best next action: $\arg\max_a Q(s', a; \theta)$
- Use **target Q-network** to evaluate that action: $Q(s', \arg\max_a Q(s', a; \theta); \theta^-)$
- This reduces overestimation bias

#### State Space Definition

**Standard DQN & Domain Gen DQN** (6D state):
```
State = [battery_0, battery_1, battery_2, battery_3, battery_4, battery_5]
```
- Normalized to [0, 1] range (divide by 100)
- Only observable beacon battery levels

**Meta-RL with MAML** (15D state):
```
State = [
    battery_0, ..., battery_5,                    # 6 values: Battery levels
    distance_0, ..., distance_5,                  # 6 values: Estimated distances
    los_flag_0, ..., los_flag_5                   # 3 values: LoS/NLoS flags
]
```
- Rich observation for quick meta-learning adaptation
- Includes distance measurements and line-of-sight conditions

**RL² LSTM** (6D state):
```
State = [battery_0, battery_1, battery_2, battery_3, battery_4, battery_5]
```
- Minimal observation - network learns dynamics implicitly via LSTM
- Challenge: Infer geometry, LoS conditions from battery patterns and hidden state

#### Reward Function (Shared Across All Models)

**Multi-objective reward**:
$$R(s_t, a_t) = 0.6 \cdot G(s_t, a_t) - 0.3 \cdot B(s_t, a_t) - 0.1 \cdot L(s_t, a_t) - 0.1 \cdot D(s_t, a_t)$$

**Components**:

1. **Geometry Quality** (weight=0.6):
   - Measures triangle area and beacon separation
   - Formula: $1 - \frac{\text{normalized_area}}{\text{max_pairwise_distance}^2}$

2. **Battery Penalty** (weight=0.3):
   - Encourages balanced selection
   - Formula: $\frac{\sigma(\text{selected_battery})}{\mu(\text{selected_battery})}$
   - Penalizes selecting nearly-depleted beacons

3. **LoS/NLoS Penalty** (weight=0.1):
   - NLoS measurements less reliable
   - Penalty: 0.2 per NLoS measurement

4. **GDOP Penalty** (weight=0.1):
   - Geometric Dilution of Precision
   - Formula: $\min(1, \text{GDOP} / 10)$
   - Lower is better for trilateration accuracy

---

## Standard DQN

### Purpose
**Baseline model**: Train single DQN on one environment to establish performance baseline.

### Architecture

```
┌─────────────────────────────────────┐
│  STANDARD DQN ARCHITECTURE          │
├─────────────────────────────────────┤
│  Input: Battery Levels (6D)         │
│    ↓                                │
│  FC1: 6 → 64 (ReLU)                │
│    ↓                                │
│  FC2: 64 → 64 (ReLU)               │
│    ↓                                │
│  FC3: 64 → 20 (Q-values)           │
│    ↓                                │
│  Output: Q-value per action         │
└─────────────────────────────────────┘
```

### Training Configuration

**File**: `src/rl/trainer_dqn.py`

| Hyperparameter | Value | Notes |
|---|---|---|
| Episodes | 500 | Per environment |
| Max Steps/Episode | 2,000 | Early termination at battery ≤10% |
| Hidden Layer Size | 64 | 3-layer MLP |
| Learning Rate | 0.001 | Adam optimizer |
| Epsilon Decay | 0.9999 | Very slow decay for exploration |
| Replay Buffer Size | 10,000 | Store 10K transitions |
| Batch Size | 32 | Train on 32 samples per step |
| Target Update Freq | 1,000 steps | Sync every 1000 gradient updates |

### Training Loop

```python
for episode in range(num_episodes):
    # 1. Reset environment and agent
    env.reset_agent_to_random_location()
    env.reset_beacon_batteries()
    state = get_battery_levels()  # 6D
    
    for step in range(max_steps):
        # 2. Action selection (epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, 19)  # Explore
        else:
            action = argmax(Q_network(state))  # Exploit
        
        # 3. Environment step
        next_state, reward, done = env.step(action)
        
        # 4. Store transition in replay buffer
        replay_buffer.push((state, action, reward, next_state, done))
        
        # 5. Train on batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(32)
        q_values = Q_network(states)
        target_q = rewards + gamma * (1 - dones) * target_network(next_states).max(1)[0]
        loss = SmoothL1Loss(q_values, target_q)
        optimizer.step(loss)
        
        # 6. Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 7. Sync target network (every 1000 steps)
        if total_steps % 1000 == 0:
            target_network.load_state_dict(Q_network.state_dict())
```

### Expected Performance

- **Convergence**: 50-100 episodes
- **Success Rate**: ~70-80% on training environment
- **Generalization**: Poor to unseen environments (overfits to single domain)

### Checkpoint Management

```
models/dqn_env_X.pt          # Single environment DQN weights
checkpoints/meta_rl/
└── dqn_env_X_timestamp.pt   # Versioned backups
```

---

## Domain Generalization DQN

### Purpose
**Multi-environment training**: Single DQN trained across diverse domains to learn generalizable beacon selection policy.

### Environment Diversity

Rather than training on a single environment, this approach trains on **randomized configurations**:

| Dimension | Range | Purpose |
|-----------|-------|---------|
| Grid Size | 10×10 to 30×30 m | Varying spatial scales |
| Beacon Count | 6 to 12 | Different selection options |
| Beacon Positions | Random | No fixed layout assumptions |
| LoS Probability | 0.4 to 0.8 | Environmental variation |
| Battery Parameters | ±20% scaling | Hardware diversity |
| CIR Model Params | Randomized | Channel impulse response variation |

### Training Configuration

**File**: `src/rl/train_domain_generalization.py`

| Hyperparameter | Value | Notes |
|---|---|---|
| Training Environments | 40+ | Diverse randomized domains |
| Episodes | 100-200 | Per environment sampling |
| Total Episodes | ~5,000+ | Sample across all environments |
| Environment Sampling | Random | Uniform sampling from training set |
| Architecture | 3-layer MLP (64 hidden) | Same as standard DQN |
| Learning Rate | 0.001 | Adam optimizer |
| Replay Buffer Size | 10,000 | Shared across all environments |
| Batch Size | 32 | Training batch |

### Training Strategy

```python
# Configuration generation
for env_id in range(40):
    # Randomize: grid_size, beacon_count, los_probability, battery, etc.
    config = {
        'grid_size': random.randint(10, 30),
        'num_beacons': random.randint(6, 12),
        'los_prob': random.uniform(0.4, 0.8),
        'battery_scale': random.uniform(0.8, 1.2),
        'consumption_scale': random.uniform(0.5, 2.0),
        # ... more randomization
    }
    save_config(config, f"domain_generalization/config_{env_id}.json")

# Meta-training loop
for meta_episode in range(num_meta_episodes):
    # Sample random environment from training set
    env_config = random.choice(training_configs)
    env = Environment(config=env_config)
    
    # Standard DQN training on this environment
    for episode in range(episodes_per_env):
        state = env.reset()
        for step in range(max_steps):
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            replay_buffer.push((state, action, reward, next_state, done))
            
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_dqn_loss(batch)
                optimizer.step(loss)
            
            state = next_state
            if done:
                break
        
        epsilon = decay_epsilon(epsilon)
```

### Configuration Ranges

**Grid Dimensions**: Ensures agent can navigate diverse spatial scales
- Minimum: 10×10 meters
- Maximum: 30×30 meters
- Agent movement: ±0.5m per step (proportionally adjusted)

**Beacon Configuration**: Variable number and placement
- Min beacons: 6
- Max beacons: 12
- Positions: Random within grid
- Minimum triangle area: 1.0 m²

**LoS/NLoS Variation**: Environmental realism
- LoS probability: [0.4, 0.8]
- Cluster count: 2-4
- Ray count: 3-5
- NLoS bias: [0.3, 1.0]

**Battery Realism**:
- Initial: [80, 120] units (vs fixed 100)
- Consumption: [2.0, 4.0] units per packet (vs fixed 3.0)
- This forces policy to handle uncertainty

**CIR Model Parameters**: Channel impulse response randomization
- Delay spread, power decay, cluster parameters all randomized
- Trains agent to handle varying channel conditions

### Expected Performance

- **Convergence**: 500-1000 episodes
- **Training Environments**: ~70-80% success rate
- **Unseen Environments**: ~50-65% success rate
- **Generalization Gap**: ~10-20% vs training performance

### Advantages & Disadvantages

**Advantages**:
- ✓ Single model handles multiple domains
- ✓ Reduced overfitting compared to single-env DQN
- ✓ Practical for real deployments (diverse indoor spaces)

**Disadvantages**:
- ✗ No explicit adaptation mechanism
- ✗ May underfit if domains too diverse
- ✗ Risk of averaging performance (good at nothing)
- ✗ Slow convergence due to environment variability

### Checkpoint Management

```
models/dqn_domain_gen.pt     # Domain generalization model
checkpoints/domain_generalization/
├── dqn_domain_generalization_20260405_200131.pt
├── dqn_domain_generalization_20260407_111410.pt
└── training_log_*.json       # Training metrics
```

---

## Meta-RL with MAML

### Purpose
**Fast adaptation**: Train a DQN that can quickly adapt to new environments with only a few gradient steps (meta-learning).

### Core Concept

**Model-Agnostic Meta-Learning (MAML)**:
1. **Inner Loop** (task adaptation): Few gradient steps on specific environment
2. **Outer Loop** (meta-update): Update meta-parameters across task batch

**Key Insight**: Learn an initialization that is sensitive to environment changes but requires few updates.

$$\theta^* = \theta - \alpha \nabla_\theta \mathcal{L}(\theta; D_{\text{train}})$$

$$\theta_{\text{new}} = \theta - \beta \nabla_\theta \mathcal{L}(\theta^*; D_{\text{test}})$$

### Architecture

Identical to standard DQN (**3-layer MLP**), but designed for parameter cloning:

```
┌─────────────────────────────────────┐
│  META-DQN ARCHITECTURE              │
├─────────────────────────────────────┤
│  Input: Full State (15D)            │
│    Battery + Distance + LoS         │
│    ↓                                │
│  FC1: 15 → 64 (ReLU)               │
│    ↓                                │
│  FC2: 64 → 64 (ReLU)               │
│    ↓                                │
│  FC3: 64 → 20 (Q-values)           │
│    ↓                                │
│  Output: Q-value per action         │
└─────────────────────────────────────┘
```

**State Dimension**: **15D** (vs 6D for standard DQN)
- Battery levels (6): $[b_0, ..., b_5]$
- Distance estimates (6): $[d_0, ..., d_5]$ (noisy measurements)
- LoS flags (3): $[\ell_0, ..., \ell_5]$ (binary indicators)

### Training Configuration

**File**: `src/rl/train_meta_rl.py`

| Hyperparameter | Value | Notes |
|---|---|---|
| **Inner Loop Steps** | 5-10 | Gradient steps per task adaptation |
| **Inner Learning Rate** | 0.01 | Smaller for quick adaptation |
| **Outer Learning Rate** | 0.001 | Meta-update rate |
| **Task Batch Size** | 4-8 | Number of tasks per meta-update |
| **Inner Buffer Size** | 500 | Transitions per task |
| **Meta Episodes** | 200-300 | Meta-training iterations |
| **Second-Order Gradients** | Yes | Compute Hessian (higher-order) |

### MAML Algorithm

**Inner-loop function** (task-specific adaptation):

```python
def inner_update(meta_model, task_env, inner_lr=0.01, inner_steps=5):
    """Adapt model to single environment task with few gradient steps."""
    
    # Clone model for this task
    adapted_model = copy.deepcopy(meta_model)
    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    # Collect data from task environment
    replay_buffer = ReplayBuffer()
    for episode in range(inner_episodes):
        state = task_env.reset()
        for step in range(max_steps_per_episode):
            action = adapted_model.select_action(state)  # Greedy
            next_state, reward, done = task_env.step(action)
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
    
    # Gradient steps on task data
    for _ in range(inner_steps):
        batch = replay_buffer.sample(batch_size=32)
        loss = compute_dqn_loss(batch, adapted_model)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    
    return adapted_model, replay_buffer
```

**Outer-loop function** (meta-parameter update):

```python
def meta_update(meta_model, task_batch, meta_lr=0.001):
    """Update meta-parameters across task batch."""
    
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    total_meta_loss = 0
    
    for task in task_batch:
        # 1. Inner-loop adaptation on this task
        adapted_model, train_buffer = inner_update(
            meta_model, 
            task.train_env,
            inner_lr=0.01,
            inner_steps=5
        )
        
        # 2. Evaluate on held-out test data from same task
        test_buffer = ReplayBuffer()
        # ... collect test data from same environment ...
        test_batch = test_buffer.sample(batch_size=32)
        
        # 3. Compute loss on test set with adapted model
        test_loss = compute_dqn_loss(test_batch, adapted_model)
        total_meta_loss += test_loss
    
    # 4. Update meta-parameters via gradient of meta-loss
    meta_optimizer.zero_grad()
    total_meta_loss.backward()  # Second-order gradients!
    meta_optimizer.step()
    
    return meta_model
```

**Full meta-training loop**:

```python
for meta_episode in range(num_meta_episodes):
    # Sample task batch (diverse environments)
    task_batch = sample_tasks(num_tasks=4)
    
    # Meta-update
    meta_model = meta_update(meta_model, task_batch, meta_lr=0.001)
    
    # Periodic evaluation
    if meta_episode % 10 == 0:
        eval_performance = evaluate_on_test_tasks(meta_model)
        print(f"Meta-episode {meta_episode}: Test performance = {eval_performance:.3f}")
```

### Test-Time Adaptation

When encountering a new (unseen) environment:

```python
def adapt_to_new_environment(trained_meta_model, new_env, num_adaptation_steps=10):
    """Quickly adapt meta-model to new environment."""
    
    # Clone meta-model
    adapted_model = copy.deepcopy(trained_meta_model)
    adapt_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
    
    # Collect data from new environment
    for episode in range(num_adaptation_episodes):
        state = new_env.reset()
        for step in range(max_steps):
            action = adapted_model.select_action(state)
            next_state, reward, done = new_env.step(action)
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
    
    # Few gradient steps (5-10)
    for _ in range(num_adaptation_steps):
        batch = replay_buffer.sample(batch_size=32)
        loss = compute_dqn_loss(batch, adapted_model)
        adapt_optimizer.zero_grad()
        loss.backward()
        adapt_optimizer.step()
    
    return adapted_model  # Now specialized for new_env
```

### Computational Cost

- **Training time**: 2-3x longer than standard DQN (due to second-order gradients)
- **Memory**: Higher (need to store gradients for meta-update)
- **Per-task adaptation**: ~1-5 seconds (only few gradient steps)

### Expected Performance

- **Warm-start**: 5-15% performance improvement after 5 adaptation steps
- **Full adaptation**: 70-85% performance after 10 adaptation steps
- **Test generalization**: ~65-75% success rate

### Checkpoint Management

```
checkpoints/meta_rl/
├── meta_dqn_10_20260407_131408.pt     # k=5 adaptation steps
├── meta_dqn_10_20260407_133109.pt     # k=5 adaptation steps
├── meta_dqn_20_20260407_131410.pt     # k=10 adaptation steps
└── ...
```

---

## RL² with LSTM

### Purpose
**Task-adaptive recurrent learning**: Use LSTM hidden state to encode task-specific information. Enables zero-shot transfer (no gradient updates at test time).

### Key Innovations

1. **Implicit Task Encoding**: LSTM hidden state carries environment dynamics
   - Beacon geometry
   - LoS/NLoS conditions
   - Agent trajectory patterns

2. **Sequence-based Training**: Train on episode sequences, not individual transitions

3. **No Test-Time Adaptation**: Only forward passes during evaluation
   - Hidden state automatically encodes new environment
   - No gradient updates needed

### Architecture

**LSTM-based Q-network**:

```
┌─────────────────────────────────────────────┐
│  RL² LSTM-DQN ARCHITECTURE                  │
├─────────────────────────────────────────────┤
│  Input: Battery Levels Sequence (6D)        │
│    [b₀, b₁, b₂, b₃, b₄, b₅]              │
│    ↓                                        │
│  LSTM Layer (128 hidden, 1 layer)          │
│    Hidden State persists within episode     │
│    Resets between episodes                  │
│    ↓                                        │
│  FC1: 128 → 128 (ReLU)                    │
│    ↓                                        │
│  FC2: 128 → 20 (Q-values)                 │
│    ↓                                        │
│  Output: Q-value per action                │
└─────────────────────────────────────────────┘
```

**Component Details**:

| Component | Configuration | Notes |
|-----------|---|---|
| **Input Size** | 6 | Battery levels only |
| **LSTM Hidden Size** | 128 | Task representation |
| **LSTM Layers** | 1 | Single recurrent layer |
| **FC1** | 128 → 128 | Transition layer |
| **FC2** | 128 → 20 | Q-value output |
| **Total Params** | ~3,000 | Slightly larger than MLP |

### Training Configuration

**File**: `src/rl/train_rl2_lstm.py`

| Hyperparameter | Value | Notes |
|---|---|---|
| **Sequence Length** | 20 | Transitions per sequence |
| **Buffer Size** | 1,000 | Number of sequences |
| **Batch Size** | 32 | Sequences per training step |
| **LSTM Hidden Size** | 128 | Task encoding dimension |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Epsilon Start** | 1.0 | Full exploration |
| **Epsilon End** | 0.05 | Minimum exploration |
| **Epsilon Decay** | 0.995 | Decay per episode |
| **Target Update Freq** | 1,000 steps | Sync target network |
| **Max Episode Length** | 150 | Steps before termination |

### Training Data Structure

**Named Tuple**:
```python
Transition = namedtuple(
    'Transition',
    ('states', 'actions', 'rewards', 'next_states', 'dones')
)

# Each element has shape: (sequence_length,)
# Example:
sequence = Transition(
    states=torch.randn(20, 6),           # 20 timesteps, 6D state
    actions=torch.randint(0, 20, (20,)), # 20 actions
    rewards=torch.randn(20),             # 20 rewards
    next_states=torch.randn(20, 6),      # 20 next states
    dones=torch.zeros(20, dtype=bool)    # 20 done flags
)
```

### Training Loop

```python
def train_rl2_lstm(num_episodes=1000):
    """Train RL² agent for multiple episodes."""
    
    model = LSTM_DQN(
        input_size=6,           # Battery levels
        hidden_size=128,
        num_actions=20
    )
    target_model = LSTM_DQN(6, 128, 20)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=1000)  # Sequence buffer
    
    epsilon = 1.0
    total_steps = 0
    
    for episode in range(num_episodes):
        # Create new environment for this episode
        env = Environment()
        env.reset_agent_to_random_location()
        
        # Collect sequence
        sequence = collect_sequence(
            env,
            model,
            epsilon,
            max_length=150
        )
        
        # Store sequence in buffer
        replay_buffer.push(sequence)
        
        # Train on batch of sequences
        if len(replay_buffer) >= BATCH_SIZE:
            sequences = replay_buffer.sample(BATCH_SIZE)
            
            for seq in sequences:
                # Forward pass through LSTM
                states = seq.states.unsqueeze(0)  # Add batch dim
                q_values, _ = model(states)       # (1, seq_len, 20)
                
                # Target Q-values
                next_states = seq.next_states.unsqueeze(0)
                target_q_vals, _ = target_model(next_states)  # (1, seq_len, 20)
                
                # Compute targets
                max_next_q = torch.max(target_q_vals, dim=2)[0].squeeze(0)  # (seq_len,)
                targets = seq.rewards + GAMMA * (1 - seq.dones) * max_next_q
                
                # Select Q-values for taken actions
                selected_q = q_values.squeeze(0)[range(len(seq.actions)), seq.actions]  # (seq_len,)
                
                # MSE loss
                loss = ((selected_q - targets) ** 2).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Sync target network
        total_steps += len(sequence)
        if total_steps % 1000 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(0.05, epsilon * 0.995)
```

### Sequence Collection

```python
def collect_sequence(env, model, epsilon, max_length=150):
    """Collect one episode as sequence of transitions."""
    
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    state = env.reset()
    hidden_state = None  # LSTM hidden state
    
    for step in range(max_length):
        # Select action
        if random.random() < epsilon:
            action = random.randint(0, 19)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)  # (1, 1, 6)
            with torch.no_grad():
                q_values, hidden_state = model(state_tensor, hidden_state)
            action = q_values.argmax(dim=2).item()
        
        # Environment step
        next_state, reward, done = env.step(action)
        
        # Store
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        if done:
            break
    
    # Convert to tensors
    return Transition(
        states=torch.tensor(states, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        next_states=torch.tensor(next_states, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.float32)
    )
```

### Test-Time Behavior

```python
def evaluate_rl2_on_new_environment(trained_model, test_env, num_episodes=10):
    """Evaluate on new environment (no gradient updates!)."""
    
    total_reward = 0
    
    for episode in range(num_episodes):
        state = test_env.reset()
        hidden_state = None  # Reset for new episode
        episode_reward = 0
        
        for step in range(150):
            # Forward pass only - use learned LSTM
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values, hidden_state = trained_model(state_tensor, hidden_state)
            
            # Greedy action selection
            action = q_values.argmax(dim=2).item()
            
            # Environment step
            next_state, reward, done = test_env.step(action)
            episode_reward += reward
            
            state = next_state
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes
```

**Key Point**: Hidden state persists within episode but resets between episodes!

### Hidden State as Task Encoder

The LSTM learns to encode:
- **Beacon Geometry**: Position/distance patterns → beacon spread inference
- **LoS Conditions**: Measurement noise patterns → LoS/NLoS prediction
- **Agent Trajectory**: Movement patterns → position estimation
- **Battery Depletion**: Depletion rates → selection policy quality

Example trajectory:
```
Episode 1 (New Env A):
  Step 0: hidden_state = [0, 0, ..., 0]              (initialized)
  Step 1: hidden_state = f_LSTM(battery, h_0)        (learns env A)
  Step 2: hidden_state = f_LSTM(battery, h_1)        (refines model)
  ...
  Step 20: hidden_state = [environment A encoding]

Episode 2 (Different Env B):
  Step 0: hidden_state = [0, 0, ..., 0]              (reset!)
  Step 1: hidden_state = f_LSTM(battery, h_0)        (learns env B)
  ...
```

### Expected Performance

- **Convergence**: 500-1000 episodes
- **Training Performance**: 75-85% success rate
- **Zero-shot Transfer**: 50-60% on new environments
- **Advantage**: No test-time adaptation needed

### Checkpoint Management

```
models/rl2_lstm.pt
checkpoints/rl2_lstm/
├── rl2_lstm_20260407_131408.pt
├── rl2_lstm_20260421_015512.pt
└── training_log_*.json
```

---

## Training Environments

### Dataset Configuration

**File**: `data/train_configs.json` (40 environments)

Standard training uses 10×10 meter grid with:
- **Beacon Count**: 6 (fixed)
- **Beacon Positions** (fixed corners/edges):
  - Corners: (1, 1), (9, 9), (1, 9), (9, 1)
  - Edges: (5, 1), (5, 9)
- **LoS Probability**: 0.5 (default)
- **Initial Battery**: 100% per beacon
- **Battery Consumption**: 3.0 units per packet

### Test Configuration

**File**: `data/test_configs.json` (10 environments)

Unseen test environments with variations:
- Grid sizes: 10×10 to 20×20
- Beacon positions: Random within grid
- LoS probability: 0.3 to 0.7
- Randomized CIR parameters

### Domain Generalization Training Set

**File**: `data/domain_generalization/` (40-50 configs)

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Grid Size | 10-30 m | Spatial scale variation |
| Beacon Count | 6-12 | Selection complexity |
| LoS Probability | 0.4-0.8 | Environmental diversity |
| Battery Init | 80-120% | Hardware randomness |
| Consumption | 0.5-2x | Energy model variation |
| CIR Parameters | Randomized | Channel diversity |

---

## Comparison of Approaches

### Summary Table

| Aspect | Standard DQN | Domain Gen | Meta-RL | RL² LSTM |
|--------|------------|-----------|---------|----------|
| **Architecture** | 3-layer MLP | 3-layer MLP | 3-layer MLP | LSTM + MLP |
| **State Size** | 6D | 6D | 15D | 6D |
| **Training Envs** | 1 | 40+ | 4-8 batch | 1 per episode |
| **Episodes** | 500 | 5,000+ | 200-300 | 1,000+ |
| **Adaptation** | None | None | Gradient-based | Recurrent |
| **Test Gradient** | No | No | Yes (5-10 steps) | No |
| **Training Time** | 1x | 5-10x | 2-3x | 3-5x |
| **Inference Speed** | Fast | Fast | Slow (5 steps) | Fast |
| **Memory** | Minimal | Moderate | High (gradients) | Moderate (hidden state) |
| **Transfer Performance** | ~40% | ~55% | ~75% | ~60% |
| **Generalization Gap** | Large | Moderate | Small | Small |

### Performance Metrics

**Training Convergence** (episodes to 70% success):
- Standard DQN: ~100 episodes
- Domain Gen: ~1,000 episodes
- Meta-RL: ~150 meta-episodes
- RL² LSTM: ~600 episodes

**Unseen Environment Success** (after deployment):
- Standard DQN: ~40%
- Domain Gen: ~55%
- Meta-RL (no adaptation): ~50%, (with 5 steps): ~75%
- RL² LSTM: ~60%

### When to Use Each

| Scenario | Recommended |
|----------|------------|
| Fast training, single environment | Standard DQN |
| Deploy to diverse buildings | Domain Gen or RL² |
| Adaptive capability needed | Meta-RL |
| Real-time constraints | Standard DQN or RL² |
| Computational budget available | Meta-RL |

---

## References & Resources

- **DQN**: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- **Double DQN**: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)
- **RL²**: Duan et al., "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning" (2016)

---

**Document Version**: 1.0  
**Last Updated**: April 29, 2026  
**Project Status**: Active Development
