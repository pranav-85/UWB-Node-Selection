import numpy as np
from typing import Tuple


class Agent:
    """Agent that moves using a random walk model."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, step_size: float = 1.0):
        """
        Initialize an Agent with starting position.
        
        Args:
            x: Initial x coordinate
            y: Initial y coordinate
            step_size: Maximum distance to move in each direction per timestep
        """
        self.x = x
        self.y = y
        self.step_size = step_size
        self.position_history = [(self.x, self.y)]
    
    def step(self) -> Tuple[float, float]:
        """
        Move the agent one timestep using random walk.
        
        Returns:
            Tuple of (new_x, new_y) coordinates
        """
        # Random walk: uniform random displacement in x and y
        dx = np.random.uniform(-self.step_size, self.step_size)
        dy = np.random.uniform(-self.step_size, self.step_size)
        
        self.x += dx
        self.y += dy
        
        self.position_history.append((self.x, self.y))
        
        return (self.x, self.y)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.x, self.y)
    
    def reset(self, x: float = 0.0, y: float = 0.0) -> None:
        """Reset agent to specified position."""
        self.x = x
        self.y = y
        self.position_history = [(self.x, self.y)]
