import numpy as np
from typing import Tuple, List


class Agent:
    """Agent that moves using a random walk model with grid boundaries and beacon collision avoidance."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, step_size: float = 1.0, 
                 grid_size: float = 10.0, beacon_positions: List[Tuple[float, float]] = None,
                 collision_radius: float = 0.5):
        """
        Initialize an Agent with starting position.
        
        Args:
            x: Initial x coordinate
            y: Initial y coordinate
            step_size: Maximum distance to move in each direction per timestep
            grid_size: Size of the grid (agent stays within [0, grid_size])
            beacon_positions: List of (x, y) positions to avoid
            collision_radius: Minimum distance to maintain from beacons
        """
        self.x = x
        self.y = y
        self.step_size = step_size
        self.grid_size = grid_size
        self.beacon_positions = beacon_positions if beacon_positions is not None else []
        self.collision_radius = collision_radius
        self.position_history = [(self.x, self.y)]
    
    def _check_beacon_collision(self, x: float, y: float) -> bool:
        """
        Check if position (x, y) is too close to any beacon.
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
        
        Returns:
            True if collision detected, False otherwise
        """
        for bx, by in self.beacon_positions:
            distance = np.sqrt((x - bx)**2 + (y - by)**2)
            if distance < self.collision_radius:
                return True
        return False
    
    def step(self) -> Tuple[float, float]:
        """
        Move the agent one timestep using random walk with boundary and collision constraints.
        
        Returns:
            Tuple of (new_x, new_y) coordinates
        """
        # Random walk: uniform random displacement in x and y
        dx = np.random.uniform(-self.step_size, self.step_size)
        dy = np.random.uniform(-self.step_size, self.step_size)
        
        # Calculate new position
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Clamp to grid boundaries
        new_x = np.clip(new_x, 0, self.grid_size)
        new_y = np.clip(new_y, 0, self.grid_size)
        
        # Check for beacon collision and adjust if necessary
        if self.beacon_positions and self._check_beacon_collision(new_x, new_y):
            # If collision detected, try to move in a different direction
            # Retry with a different random move
            max_retries = 5
            for _ in range(max_retries):
                dx = np.random.uniform(-self.step_size, self.step_size)
                dy = np.random.uniform(-self.step_size, self.step_size)
                new_x = np.clip(self.x + dx, 0, self.grid_size)
                new_y = np.clip(self.y + dy, 0, self.grid_size)
                
                if not self._check_beacon_collision(new_x, new_y):
                    break
            else:
                # If all retries fail, stay in current position
                new_x = self.x
                new_y = self.y
        
        self.x = new_x
        self.y = new_y
        self.position_history.append((self.x, self.y))
        
        return (self.x, self.y)
    
    def move(self, direction: str) -> Tuple[float, float]:
        """
        Move the agent in a specified direction with boundary and collision constraints.
        
        Args:
            direction: Direction to move ('up', 'down', 'left', 'right')
        
        Returns:
            Tuple of (new_x, new_y) coordinates
        """
        # Calculate displacement based on direction
        dx = 0.0
        dy = 0.0
        
        if direction == 'up':
            dy = self.step_size
        elif direction == 'down':
            dy = -self.step_size
        elif direction == 'left':
            dx = -self.step_size
        elif direction == 'right':
            dx = self.step_size
        
        # Calculate new position
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Clamp to grid boundaries
        new_x = np.clip(new_x, 0, self.grid_size)
        new_y = np.clip(new_y, 0, self.grid_size)
        
        # Check for beacon collision and adjust if necessary
        if self.beacon_positions and self._check_beacon_collision(new_x, new_y):
            # If collision detected, stay in current position
            new_x = self.x
            new_y = self.y
        
        self.x = new_x
        self.y = new_y
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
    
    def set_beacon_positions(self, beacon_positions: List[Tuple[float, float]]) -> None:
        """Update beacon positions for collision avoidance."""
        self.beacon_positions = beacon_positions
