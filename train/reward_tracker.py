import numpy as np
from typing import Dict, List, Any

class RewardTracker:
    """
    Tracks and computes rewards for the P2P Energy Market.
    Separates financial (profit), environmental (grid penalty), and operational (SoC, battery) components.
    
    Now supports Phase 5 Hybrid Agent Features:
    - V2G Bonus
    - 3x Grid Peak Penalty
    - Action Smoothing
    """
    
    def __init__(self, n_agents: int, fairness_coeff: float = 0.5):
        self.n_agents = n_agents
        self.fairness_coeff = fairness_coeff
        self.reset()

    def reset(self):
        """Resets the internal tracking stats."""
        # Current timestep stats
        self.step_profit = np.zeros(self.n_agents)
        self.step_grid_penalty = np.zeros(self.n_agents) 
        self.step_soc_penalty = np.zeros(self.n_agents)
        self.step_grid_overload_penalty = np.zeros(self.n_agents) 
        self.step_battery_cost = np.zeros(self.n_agents)
        self.step_fairness_penalty = 0.0
        
        # Cumulative stats (optional, usually handled by wrapper/logger)
        self.history: List[Dict[str, Any]] = []

    def compute_gini(self, x: np.ndarray) -> float:
        """
        Compute Gini coefficient for 1D array x.
        """
        x = np.array(x, dtype=float)
        if x.size == 0:
            return 0.0
        
        # Shift to ensure non-negative for standard Gini calculation
        if np.min(x) < 0:
            x -= np.min(x)
            
        # Handle all zeros case
        if np.sum(x) == 0:
            return 0.0
            
        x_sorted = np.sort(x)
        n = x_sorted.size
        i = np.arange(1, n + 1)
        numerator = np.sum((2 * i - n - 1) * x_sorted)
        denominator = n * np.sum(x_sorted)
        
        return float(numerator / denominator)

    def calculate_total_reward(self, 
                             profits: np.ndarray, 
                             grid_import_penalties: np.ndarray,
                             soc_penalties: np.ndarray,
                             grid_overload_costs: np.ndarray,
                             battery_costs: np.ndarray,
                             total_export_kw: float = 0.0) -> float:
        """
        Aggregates the 5 core components into a scalar reward.
        Rationale for simplification:
        1. Profit: Core objective (maximize revenue/minimize cost).
        2. Grid Penalty: Discourage reliance on external grid.
        3. SoC Penalty: Keep batteries healthy (close to target).
        4. Battery Deg: Penalize excessive cycling.
        5. Fairness: Encourage equitable P2P trading.
        """
        
        # 1. Update internal state with Clipping
        self.step_profit = profits
        self.step_grid_penalty = np.clip(grid_import_penalties, 0.0, 50.0) 
        self.step_soc_penalty = np.clip(soc_penalties, 0.0, 10.0)
        self.step_grid_overload_penalty = np.clip(grid_overload_costs, 0.0, 10.0)
        self.step_battery_cost = np.clip(battery_costs, 0.0, 50.0)
        
        # 2. Compute per-agent net value (before fairness)
        # REWARD FORMULA: Profit - Costs - Penalties
        per_agent_net = (profits 
                         - self.step_grid_penalty 
                         - self.step_soc_penalty 
                         - self.step_grid_overload_penalty 
                         - self.step_battery_cost)
        
        # 3. Compute Fairness Penalty (Global)
        gini = self.compute_gini(per_agent_net)
        self.step_fairness_penalty = self.fairness_coeff * gini * (1.0 + 0.05 * abs(total_export_kw))
        
        # 4. Total Shared Reward
        total_reward = np.sum(per_agent_net) - self.step_fairness_penalty
        
        return float(total_reward)



    def get_info(self) -> Dict[str, Any]:
        """Returns dict of current step stats for logging."""
        return {
            "mean_profit": np.mean(self.step_profit),
            "mean_grid_penalty": np.mean(self.step_grid_penalty),
            "mean_soc_penalty": np.mean(self.step_soc_penalty),
            "mean_grid_overload_penalty": np.mean(self.step_grid_overload_penalty),
            "mean_battery_cost": np.mean(self.step_battery_cost),
            "fairness_penalty": self.step_fairness_penalty
        }
