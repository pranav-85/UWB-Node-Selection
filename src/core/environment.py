import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
from pathlib import Path
import csv
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.agent import Agent
from models.beacon import Beacon, UWBHardwareParams
from core.link_model import assign_beacon_links
from config import (
    GRID_SIZE, AGENT_INITIAL_X, AGENT_INITIAL_Y, AGENT_STEP_SIZE,
    NUM_BEACONS, BEACON_INITIAL_BATTERY, NUM_SELECTED_BEACONS, LOS_PROBABILITY
)


class Environment:
    """10x10 grid environment with beacons and a mobile agent."""
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """
        Initialize the environment.
        
        Args:
            grid_size: Size of the grid (default from config)
        """
        self.grid_size = grid_size
        
        # Create 6 beacons at corners and near walls
        self.beacons = self._create_beacons()
        
        # Create one mobile agent at center
        self.agent = Agent(x=AGENT_INITIAL_X, y=AGENT_INITIAL_Y, step_size=AGENT_STEP_SIZE)
        
        # Network graph
        self.graph = None
        self.current_links = None
        self.selected_beacon_indices = []  # Track which beacons are selected for localization
        
        # Recording data
        self.recording = False
        self.records = []  # List to store timestep records
        
    def _create_beacons(self) -> List[Beacon]:
        """Create 6 beacons at corners and near walls."""
        positions = [
            (0.5, 0.5),              # Bottom-left corner
            (self.grid_size - 0.5, 0.5),    # Bottom-right corner
            (0.5, self.grid_size - 0.5),    # Top-left corner
            (self.grid_size - 0.5, self.grid_size - 0.5),  # Top-right corner
            (self.grid_size / 2, 0.5),      # Bottom wall middle
            (self.grid_size / 2, self.grid_size - 0.5),    # Top wall middle
        ]
        
        uwb_params = UWBHardwareParams()
        beacons = [
            Beacon(beacon_id=i, position=pos, uwb_params=uwb_params, initial_battery=BEACON_INITIAL_BATTERY)
            for i, pos in enumerate(positions)
        ]
        return beacons
    
    def step(self):
        """Move agent and assign new links and consume energy from selected beacons."""
        self.agent.step()
        self._assign_links()
        
        # Select random beacons for localization
        self.selected_beacon_indices = list(np.random.choice(len(self.beacons), size=NUM_SELECTED_BEACONS, replace=False))
        
        # Consume energy only from selected beacons
        for idx in self.selected_beacon_indices:
            self.beacons[idx].use_for_localization()
        
        # Record if recording is enabled
        if self.recording:
            self._record_step()
    
    def _assign_links(self):
        """Assign LoS/NLoS links to each beacon."""
        self.current_links = assign_beacon_links(len(self.beacons))
    
    def start_recording(self):
        """Start recording the simulation."""
        self.recording = True
        self.records = []
    
    def stop_recording(self):
        """Stop recording the simulation."""
        self.recording = False
    
    def _record_step(self):
        """Record current step data."""
        agent_x, agent_y = self.agent.get_position()
        selected_beacons_str = ','.join(str(idx) for idx in self.selected_beacon_indices)
        los_links_str = ','.join(str(int(link)) for link in self.current_links)
        
        self.records.append({
            'timestep': len(self.records),
            'agent_x': agent_x,
            'agent_y': agent_y,
            'selected_beacons': selected_beacons_str,
            'los_links': los_links_str
        })
    
    def save_scenario(self, filename: str = None):
        """
        Save recorded scenario to CSV file.
        
        Args:
            filename: Name of the file (without extension). If None, uses timestamp.
        """
        if not self.records:
            print("No records to save. Start recording first.")
            return
        
        scenarios_dir = Path(__file__).parent.parent / 'scenarios'
        scenarios_dir.mkdir(exist_ok=True)
        
        if filename is None:
            filename = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = scenarios_dir / f"{filename}.csv"
        
        # Write to CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestep', 'agent_x', 'agent_y', 'selected_beacons', 'los_links']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(self.records)
        
        print(f"Scenario saved to {filepath}")
    
    def visualize(
        self,
        title: str = "UWB Network Simulation",
        ax=None,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """
        Visualize the environment with beacons, agent, and network links.

        Args:
            title: Title for the visualization
            ax: Existing matplotlib Axes to draw on (for live updates)
            figsize: Figure size as (width, height)
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            ax.clear()

        
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        
        self.graph = nx.Graph()

        # Add beacon nodes
        for i, beacon in enumerate(self.beacons):
            self.graph.add_node(f'B{i}', pos=beacon.position)

        # Add agent node
        agent_pos = self.agent.get_position()
        self.graph.add_node('Agent', pos=agent_pos)

        # Assign links if not already assigned
        if self.current_links is None:
            self._assign_links()

        # Add edges with LoS / NLoS distinction
        for i, link_type in enumerate(self.current_links):
            self.graph.add_edge(f'B{i}', 'Agent', link_type=link_type)

        
        pos = nx.get_node_attributes(self.graph, 'pos')

        los_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['link_type'] == 1]
        nlos_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['link_type'] == 0]

        nx.draw_networkx_edges(
            self.graph, pos, edgelist=los_edges,
            edge_color='green', width=2, alpha=0.7, ax=ax, label='LoS'
        )

        nx.draw_networkx_edges(
            self.graph, pos, edgelist=nlos_edges,
            edge_color='red', width=2, alpha=0.7, ax=ax, label='NLoS'
        )

        beacon_nodes = [f'B{i}' for i in range(len(self.beacons))]
        selected_beacon_nodes = [f'B{i}' for i in self.selected_beacon_indices]
        non_selected_beacon_nodes = [node for node in beacon_nodes if node not in selected_beacon_nodes]
        
        # Draw non-selected beacons in blue
        nx.draw_networkx_nodes(
            self.graph, pos, nodelist=non_selected_beacon_nodes,
            node_color='blue', node_size=300, ax=ax, label='Beacons'
        )
        
        # Draw selected beacons in red (highlighted)
        nx.draw_networkx_nodes(
            self.graph, pos, nodelist=selected_beacon_nodes,
            node_color='red', node_size=400, ax=ax, label='Selected Beacons'
        )

        nx.draw_networkx_nodes(
            self.graph, pos, nodelist=['Agent'],
            node_color='orange', node_size=500, ax=ax, label='Agent'
        )

        nx.draw_networkx_labels(
            self.graph, pos, font_size=8, font_weight='bold', ax=ax
        )

        # Add battery percentage labels on top of beacons
        for i, beacon in enumerate(self.beacons):
            battery_text = f"{beacon.current_battery_level():.1f}%"
            ax.text(beacon.position[0], beacon.position[1] + 0.5, battery_text, 
                   ha='center', va='bottom', fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend(loc='upper left')

        fig.tight_layout()

        return fig, ax

    
    def reset(self):
        """Reset the environment."""
        self.agent.reset(x=AGENT_INITIAL_X, y=AGENT_INITIAL_Y)
        self.current_links = None
        self.selected_beacon_indices = []
