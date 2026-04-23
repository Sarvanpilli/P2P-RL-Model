import numpy as np
from typing import Dict, List, Any

class RewardTracker:
    """
    Tracks and computes rewards for the P2P Energy Market.
    Simplified SLIM v7: Focuses on Breaking Battery-Only Local Optimum.
    """
    
    def __init__(self, n_agents: int, fairness_coeff: float = 0.5):
        self.n_agents = n_agents
        self.fairness_coeff = fairness_coeff
        self.reset()

    def reset(self):
        """Resets the internal tracking stats."""
        self.step_p2p_revenue = np.zeros(self.n_agents)
        self.step_grid_cost = np.zeros(self.n_agents)
        self.step_clean_profit = np.zeros(self.n_agents)
        self.step_battery_penalty = np.zeros(self.n_agents)
        self.step_missed_trade_penalty = 0.0
        self.step_p2p_bonus = np.zeros(self.n_agents)
        self.step_trade_success_bonus = 0.0
        self.step_fairness_penalty = 0.0
        self.history: List[Dict[str, Any]] = []

    def compute_gini(self, x: np.ndarray) -> float:
        x = np.array(x, dtype=float)
        if x.size == 0 or np.sum(x) == 0: return 0.0
        if np.min(x) < 0: x -= np.min(x)
        x_sorted = np.sort(x)
        n = x_sorted.size
        i = np.arange(1, n + 1)
        numerator = np.sum((2 * i - n - 1) * x_sorted)
        denominator = n * np.sum(x_sorted)
        return float(numerator / denominator)

    def calculate_total_reward(self, 
                             p2p_revenue: np.ndarray,
                             p2p_cost: np.ndarray,
                             grid_cost: np.ndarray,
                             traded_energy: np.ndarray,
                             avg_demand: np.ndarray,
                             battery_throughput: np.ndarray,
                             trade_matched: np.ndarray,
                             trade_possible: bool,
                             battery_used_step: bool,
                             alpha: float = 0.10,        # Boosted for v7 emergence
                             beta: float = 1.5,         # Grid import penalty
                             gamma_batt: float = 0.015,   # Battery discharge penalty
                             lambda_missed: float = 0.05  # Missed trade penalty
                             ) -> np.ndarray:
        # --- Step 1: Economic Core ---
        self.step_p2p_revenue = p2p_revenue
        self.step_p2p_cost = p2p_cost
        self.step_grid_cost = grid_cost
        self.step_clean_profit = p2p_revenue - p2p_cost - grid_cost
        
        normalized_profit = self.step_clean_profit / (avg_demand + 1e-6)
        
        # --- Step 2: Missed Trade Penalty ---
        no_trade_occurred = np.sum(trade_matched) < 1e-6
        self.step_missed_trade_penalty = lambda_missed if (trade_possible and no_trade_occurred) else 0.0
        
        # --- Step 3: Battery Dominance Penalty ---
        self.step_battery_penalty = gamma_batt * battery_throughput
        if trade_possible and battery_used_step:
            self.step_battery_penalty += 0.03
            
        # --- Step 4: P2P Incentive ---
        self.step_p2p_bonus = alpha * traded_energy
        
        # --- Step 5: Trade Priority Signal ---
        self.step_trade_success_bonus = 0.05 if (trade_possible and not no_trade_occurred) else 0.0
        
        # Aggregation
        per_agent_reward = (
            normalized_profit
            + self.step_p2p_bonus
            - (beta * grid_cost / 0.20)
            - self.step_battery_penalty
            - self.step_missed_trade_penalty
            + self.step_trade_success_bonus
        )
        
        gini = self.compute_gini(per_agent_reward)
        self.step_fairness_penalty = 0.5 * self.fairness_coeff * gini
        
        return (per_agent_reward - self.step_fairness_penalty).astype(np.float32)

    def get_info(self) -> Dict[str, Any]:
        return {
            "mean_clean_profit": np.mean(self.step_clean_profit),
            "mean_p2p_bonus": np.mean(self.step_p2p_bonus),
            "mean_battery_penalty": np.mean(self.step_battery_penalty),
            "missed_trade_penalty": self.step_missed_trade_penalty,
            "fairness_penalty": self.step_fairness_penalty
        }
