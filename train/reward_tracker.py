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
        self.smoothing_coeff = 0.5 # New Phase 5 Param
        self.reset()

    def reset(self):
        """Resets the internal tracking stats."""
        # Current timestep stats
        self.step_profit = np.zeros(self.n_agents)
        self.step_grid_penalty = np.zeros(self.n_agents) # Now used for Import Penalty
        self.step_soc_penalty = np.zeros(self.n_agents)
        self.step_grid_overload_penalty = np.zeros(self.n_agents) # Renamed old grid penalty
        self.step_battery_cost = np.zeros(self.n_agents)
        self.step_smoothing_penalty = np.zeros(self.n_agents)
        self.step_deep_discharge_penalty = np.zeros(self.n_agents)
        self.step_export_penalty = np.zeros(self.n_agents)
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
                             export_penalties: np.ndarray,
                             smoothing_penalties: np.ndarray = None,
                             deep_discharge_penalties: np.ndarray = None,
                             v2g_bonus: np.ndarray = None,
                             total_export_kw: float = 0.0) -> float:
        """
        Aggregates all components into a scalar reward.
        """
        if smoothing_penalties is None: smoothing_penalties = np.zeros(self.n_agents)
        if deep_discharge_penalties is None: deep_discharge_penalties = np.zeros(self.n_agents)
        if v2g_bonus is None: v2g_bonus = np.zeros(self.n_agents)

        # 1. Update internal state with Clipping
        self.step_profit = profits
        self.step_grid_penalty = np.clip(grid_import_penalties, 0.0, 50.0) 
        self.step_soc_penalty = np.clip(soc_penalties, 0.0, 10.0)
        self.step_grid_overload_penalty = np.clip(grid_overload_costs, 0.0, 10.0)
        self.step_battery_cost = np.clip(battery_costs, 0.0, 50.0)
        self.step_export_penalty = np.clip(export_penalties, 0.0, 5.0)
        self.step_smoothing_penalty = np.clip(smoothing_penalties, 0.0, 10.0)
        self.step_deep_discharge_penalty = np.clip(deep_discharge_penalties, 0.0, 10.0)
        
        # 2. Compute per-agent net value (before fairness)
        # REWARD FORMULA: Profit + Bonus - Costs - Penalties
        per_agent_net = (profits 
                         + v2g_bonus
                         - self.step_grid_penalty 
                         - soc_penalties 
                         - grid_overload_costs 
                         - battery_costs 
                         - export_penalties
                         - smoothing_penalties
                         - deep_discharge_penalties)
        
        # 3. Compute Fairness Penalty (Global)
        gini = self.compute_gini(per_agent_net)
        self.step_fairness_penalty = self.fairness_coeff * gini * (1.0 + 0.05 * abs(total_export_kw))
        
        # 4. Total Shared Reward
        total_reward = np.sum(per_agent_net) - self.step_fairness_penalty
        
        return float(total_reward)

    def calculate_smoothing_penalty(self, current_action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        """
        Computes penalty for action volatility.
        R = -lambda * |a_t - a_{t-1}|
        Summed across action dimensions per agent.
        """
        # Ensure shapes match or broadcast
        # current and prev should be (N, 3)
        diff = np.abs(current_action - prev_action)
        # Sum deviation across the 3 action dimensions (Charge, Trade, Bid)
        # Result should be (N,)
        penalty = np.sum(diff, axis=1) * self.smoothing_coeff
        return penalty

    def calculate_peak_penalty(self, grid_import_kw: np.ndarray, hour: int) -> np.ndarray:
        """
        Triples the negative reward for importing from grid between 17:00 and 21:00.
        """
        penalty = np.zeros_like(grid_import_kw)
        
        # Peak Hours: 17, 18, 19, 20 (up to 21:00)
        if 17 <= hour < 21:
            # Base import cost is handled in profit (Price * Flow)
            # This is an EXTRA penalty to discourage use.
            # Grid import KW * Rate (e.g. 1.5x regular?)
            # Regular penalty is 0.5 * kW.
            # Peak penalty adds another 1.0 * kW -> Total 1.5 * kW
            # Or simplified: specific peak penalty component.
            penalty = grid_import_kw * 1.5
            
        return penalty

    def calculate_v2g_bonus(self, socs: np.ndarray, capacities: np.ndarray, hour: int) -> np.ndarray:
        """
        Agent 2 (EV) gets bonus if SOC > 80% at 7:00 AM (Hour 7).
        """
        bonus = np.zeros(self.n_agents)
        # Agent 2 is Index 2 (assuming fixed indexing for now)
        # We should probably pass agent_type or ID, but for Phase 5 hardcoded is fine.
        
        if hour == 7:
            # Check Agent 2
            ev_idx = 2
            if ev_idx < self.n_agents:
                soc_pct = socs[ev_idx] / capacities[ev_idx] if capacities[ev_idx] > 0 else 0
                if soc_pct > 0.80:
                    bonus[ev_idx] = 10.0 # Significant bonus
                    
        return bonus

    def get_info(self) -> Dict[str, Any]:
        """Returns dict of current step stats for logging."""
        return {
            "mean_profit": np.mean(self.step_profit),
            "mean_grid_penalty": np.mean(self.step_grid_penalty),
            "mean_soc_penalty": np.mean(self.step_soc_penalty),
            "mean_grid_overload_penalty": np.mean(self.step_grid_overload_penalty),
            "mean_battery_cost": np.mean(self.step_battery_cost),
            "mean_smoothing_penalty": np.mean(self.step_smoothing_penalty),
            "mean_deep_discharge_penalty": np.mean(self.step_deep_discharge_penalty),
            "fairness_penalty": self.step_fairness_penalty
        }
