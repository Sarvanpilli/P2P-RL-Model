
import numpy as np
from typing import Dict, List, Any

class RewardTracker:
    """
    Tracks and computes individual reward components for multi-agent energy environments.
    
    This class helps in modularizing the reward function and allows for detailed
    logging of what contributes to the agent's signal (Profit vs Penalty).
    """
    
    def __init__(self, n_agents: int, fairness_coeff: float = 0.5):
        self.n_agents = n_agents
        self.fairness_coeff = fairness_coeff
        self.reset()

    def reset(self):
        """Resets the internal tracking stats."""
        # Current timestep stats
        self.step_profit = np.zeros(self.n_agents)
        self.step_co2_penalty = np.zeros(self.n_agents)
        self.step_soc_penalty = np.zeros(self.n_agents)
        self.step_grid_penalty = np.zeros(self.n_agents)
        self.step_battery_cost = np.zeros(self.n_agents)
        self.step_export_penalty = np.zeros(self.n_agents)
        self.step_fairness_penalty = 0.0
        
        # Cumulative stats (optional, usually handled by wrapper/logger)
        self.history: List[Dict[str, Any]] = []

    def compute_gini(self, x: np.ndarray) -> float:
        """
        Compute Gini coefficient for 1D array x.
        
        Args:
            x: Array of values (profits/rewards). 
               Note: Gini is typically for non-negative values. 
               If x contains negatives, we shift by min to make positive.
        
        Returns:
            float: Gini coefficient [0, 1]
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
        cumx = np.cumsum(x_sorted)
        sum_x = cumx[-1]
        
        index = np.arange(1, n + 1)
        # Gini formula: (2 * sum(i * x_i) - (n + 1) * sum(x)) / (n * sum(x))
        numerator = 2.0 * np.sum(index * x_sorted) - (n + 1) * sum_x
        denominator = n * sum_x + 1e-12
        
        return float(numerator / denominator)

    def calculate_total_reward(self, 
                             profits: np.ndarray, 
                             co2_penalties: np.ndarray,
                             soc_penalties: np.ndarray,
                             grid_overload_costs: np.ndarray,
                             battery_costs: np.ndarray,
                             export_penalties: np.ndarray,
                             total_export_kw: float) -> float:
        """
        Aggregates all components into a scalar reward.
        
        Args:
            profits: (n_agents,) Net financial profit (Revenue - Import Cost)
            co2_penalties: (n_agents,) Carbon cost (positive value to be subtracted)
            soc_penalties: (n_agents,) SoC deviation/safety penalty
            grid_overload_costs: (n_agents,) Line congestion penalty
            battery_costs: (n_agents,) Degradation cost
            export_penalties: (n_agents,) Individual export limit penalty
            total_export_kw: Total grid export (used for fairness scaling)
            
        Returns:
            float: The global reward (scalar, assuming Shared PPO)
        """
        # 1. Update internal state
        # 1. Update internal state with Clipping to prevent explosion
        self.step_profit = profits
        self.step_co2_penalty = np.clip(co2_penalties, 0.0, 10.0)
        self.step_soc_penalty = np.clip(soc_penalties, 0.0, 10.0)
        self.step_grid_penalty = np.clip(grid_overload_costs, 0.0, 10.0)
        self.step_battery_cost = np.clip(battery_costs, 0.0, 50.0)
        self.step_export_penalty = np.clip(export_penalties, 0.0, 5.0)
        
        # 2. Compute per-agent net value (before fairness)
        # We subtract costs/penalties from profit
        per_agent_net = (profits 
                         - co2_penalties 
                         - soc_penalties 
                         - grid_overload_costs 
                         - battery_costs 
                         - export_penalties)
        
        # 3. Compute Fairness Penalty (Global)
        # We compute Gini on the *profits* or *net value*? 
        # Usually on profits to ensure equitable financial outcome.
        gini = self.compute_gini(per_agent_net)
        
        # Scale fairness penalty slightly by export volume to prevent it dominating 
        # low-activity periods, or use fixed coeff.
        # Logic from previous env: total_profit - fairness_coeff * gini * (1 + 0.05 * export)
        self.step_fairness_penalty = self.fairness_coeff * gini * (1.0 + 0.05 * abs(total_export_kw))
        
        # 4. Total Shared Reward
        # Sum of all agents' net value minus global fairness penalty
        total_reward = np.sum(per_agent_net) - self.step_fairness_penalty
        
        return float(total_reward)

    def get_info(self) -> Dict[str, float]:
        """Returns a dictionary of the last step's components for logging."""
        return {
            "reward/profit_mean": np.mean(self.step_profit),
            "reward/co2_penalty_mean": np.mean(self.step_co2_penalty),
            "reward/soc_penalty_mean": np.mean(self.step_soc_penalty),
            "reward/grid_penalty_mean": np.mean(self.step_grid_penalty),
            "reward/batt_cost_mean": np.mean(self.step_battery_cost),
            "reward/fairness_penalty": self.step_fairness_penalty,
            "reward/gini_index": self.compute_gini(self.step_profit) # Pure profit gini for metric
        }
