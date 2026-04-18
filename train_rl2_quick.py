#!/usr/bin/env python3
"""
Quick RL² LSTM training for beacon selection.
Trains for fewer episodes (50) to generate initial checkpoints quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rl.train_rl2_lstm import train_rl2

if __name__ == '__main__':
    # Train for 50 episodes (instead of 500) to generate compatible checkpoints quickly
    # Later can extend training if needed
    model, log = train_rl2(
        num_episodes=50,
        num_train_steps=10,
        save_freq=10,  # Save every 10 episodes
        eval_freq=10   # Evaluate every 10 episodes
    )
    
    print("\n✓ RL² LSTM training for beacon selection complete!")
    print("Checkpoints ready for evaluation.")
