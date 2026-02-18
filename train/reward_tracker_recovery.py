"""
Phase 5 Recovery: Reward Tracker with Positive Reinforcement

Key Changes from Original:
1. P2P Participation Bonus: Reward agents for trading with peers
2. Healthy SoC Bonus: Reward maintaining SoC in 30-70% range
3. Reduced Penalty Coefficients: Less aggressive penalties
4. High Unmet Demand Penalty: Ensure agents don't ignore load
5. Demand Satisfaction Bonus: Reward meeting demand efficiently

Philosophy: "Carrots over Sticks" - Guide learning with positive rewards
rather than punishing bad behavior.
"""

import numpy as np
from typing import Dict, Any


class RewardTrackerRecovery:
    """
    Recovery Reward Tracker with Positive Reinforcement.
    
    Designed to fix policy collapse by:
    - Encouraging P2P trading through bonuses
    - Rewarding healthy battery management
    - Reducing penalty magnitudes for stability
    """
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        
        # === RECOVERY PARAMETERS ===
        # Positive Reinforcement
        self.p2p_participation_bonus_coeff = 2.0      # NEW: Reward P2P trading
        self.healthy_soc_bonus_coeff = 0.1            # NEW: Reward 30-70% SoC
        self.demand_satisfaction_bonus_coeff = 1.0    # NEW: Reward meeting demand
        
        # Reduced Penalties (from original)
        self.smoothing_penalty_coeff = 0.01           # Was 0.5 -> Reduced 98%
        self.battery_degradation_coeff = 0.005        # Was 0.05 -> Reduced 90%
        self.grid_import_penalty_coeff = 0.3          # Was 0.5 -> Reduced 40%
        self.soc_deviation_penalty_coeff = 0.0001     # Was 0.001 -> Reduced 90%
        
        # High Unmet Demand Penalty (to prevent ignoring load)
        self.unmet_demand_penalty_coeff = 5.0         # NEW: Strong penalty
        
        # Fairness (keep moderate)
        self.fairness_coeff = 0.3                     # Was 0.5 -> Reduced 40%
        
        self.reset()
    
    def reset(self):
        """Reset tracking stats"""
        # Current step components
        self.step_p2p_bonus = np.zeros(self.n_agents)
        self.step_healthy_soc_bonus = np.zeros(self.n_agents)
        self.step_demand_satisfaction_bonus = np.zeros(self.n_agents)
        self.step_grid_import_penalty = np.zeros(self.n_agents)
        self.step_smoothing_penalty = np.zeros(self.n_agents)
        self.step_battery_cost = np.zeros(self.n_agents)
        self.step_soc_penalty = np.zeros(self.n_agents)
        self.step_unmet_demand_penalty = np.zeros(self.n_agents)
        self.step_grid_overload_penalty = np.zeros(self.n_agents)
        self.step_fairness_penalty = 0.0
        
        # Profit tracking
        self.step_p2p_profit = np.zeros(self.n_agents)
        self.step_grid_cost = np.zeros(self.n_agents)
    
    def calculate_total_reward(self,
                              p2p_trades_kw: np.ndarray,
                              p2p_price: float,
                              grid_flows_kw: np.ndarray,
                              grid_retail_price: float,
                              grid_feedin_price: float,
                              socs: np.ndarray,
                              capacities: np.ndarray,
                              throughputs: np.ndarray,
                              line_overload_kw: float,
                              current_action: np.ndarray,
                              prev_action: np.ndarray,
                              hour: int,
                              demands: np.ndarray) -> float:
        """
        Calculate total reward with positive reinforcement.
        
        Reward Structure:
        R = Profit + Bonuses - Penalties
        
        Bonuses (NEW):
        - P2P participation bonus
        - Healthy SoC bonus (30-70%)
        - Demand satisfaction bonus
        
        Penalties (REDUCED):
        - Grid import penalty (reduced)
        - Smoothing penalty (reduced)
        - Battery degradation (reduced)
        - Unmet demand penalty (high)
        """
        
        # === 1. PROFIT CALCULATION ===
        # P2P Profit
        self.step_p2p_profit = p2p_trades_kw * p2p_price
        
        # Grid Costs (negative for imports, positive for exports)
        grid_costs = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if grid_flows_kw[i] < 0:  # Import
                grid_costs[i] = grid_flows_kw[i] * grid_retail_price  # Negative cost
            else:  # Export
                grid_costs[i] = grid_flows_kw[i] * grid_feedin_price  # Positive revenue
        
        self.step_grid_cost = grid_costs
        
        total_profit = self.step_p2p_profit + grid_costs
        
        # === 2. BONUSES (Positive Reinforcement) ===
        
        # 2a. P2P Participation Bonus
        # Reward proportional to P2P volume traded
        p2p_volume_pct = np.abs(p2p_trades_kw) / (demands + 1e-6)  # % of demand met via P2P
        self.step_p2p_bonus = np.clip(p2p_volume_pct, 0, 1) * self.p2p_participation_bonus_coeff
        
        # 2b. Healthy SoC Bonus
        # Reward maintaining SoC in 30-70% range
        soc_pcts = socs / (capacities + 1e-6)
        healthy_mask = (soc_pcts >= 0.3) & (soc_pcts <= 0.7)
        self.step_healthy_soc_bonus = healthy_mask.astype(float) * self.healthy_soc_bonus_coeff
        
        # 2c. Demand Satisfaction Bonus
        # Reward if demand is being met (not running on empty battery)
        # Proxy: SoC > 10% means agent can meet demand
        demand_met_mask = soc_pcts > 0.10
        self.step_demand_satisfaction_bonus = demand_met_mask.astype(float) * self.demand_satisfaction_bonus_coeff
        
        # === 3. PENALTIES (Reduced Magnitudes) ===
        
        # 3a. Grid Import Penalty (Reduced)
        grid_imports_kw = np.abs(np.clip(grid_flows_kw, None, 0))
        self.step_grid_import_penalty = grid_imports_kw * self.grid_import_penalty_coeff
        
        # 3b. Smoothing Penalty (Reduced)
        action_diff = np.abs(current_action - prev_action)
        self.step_smoothing_penalty = np.sum(action_diff, axis=1) * self.smoothing_penalty_coeff
        
        # 3c. Battery Degradation Cost (Reduced)
        self.step_battery_cost = throughputs * self.battery_degradation_coeff
        
        # 3d. SoC Deviation Penalty (Reduced)
        # Penalize extreme SoC (too high or too low)
        target_soc_pct = 0.5
        soc_deviation = np.abs(soc_pcts - target_soc_pct)
        self.step_soc_penalty = (soc_deviation ** 2) * self.soc_deviation_penalty_coeff
        
        # 3e. Unmet Demand Penalty (High - Critical)
        # Strong penalty if SoC < 10% (can't meet demand)
        unmet_mask = soc_pcts < 0.10
        self.step_unmet_demand_penalty = unmet_mask.astype(float) * self.unmet_demand_penalty_coeff
        
        # 3f. Grid Overload Penalty
        overload_penalty_per_agent = (line_overload_kw * self.n_agents) / self.n_agents
        self.step_grid_overload_penalty = np.ones(self.n_agents) * overload_penalty_per_agent
        
        # === 4. PER-AGENT NET VALUE ===
        per_agent_net = (
            total_profit
            + self.step_p2p_bonus
            + self.step_healthy_soc_bonus
            + self.step_demand_satisfaction_bonus
            - self.step_grid_import_penalty
            - self.step_smoothing_penalty
            - self.step_battery_cost
            - self.step_soc_penalty
            - self.step_unmet_demand_penalty
            - self.step_grid_overload_penalty
        )
        
        # === 5. FAIRNESS PENALTY (Reduced) ===
        gini = self._compute_gini(per_agent_net)
        self.step_fairness_penalty = self.fairness_coeff * gini
        
        # === 6. TOTAL REWARD ===
        total_reward = np.sum(per_agent_net) - self.step_fairness_penalty
        
        return float(total_reward)
    
    def _compute_gini(self, x: np.ndarray) -> float:
        """Compute Gini coefficient for fairness"""
        x = np.array(x, dtype=float)
        if x.size == 0:
            return 0.0
        
        # Shift to non-negative
        if np.min(x) < 0:
            x -= np.min(x)
        
        # Handle all zeros
        if np.sum(x) == 0:
            return 0.0
        
        x_sorted = np.sort(x)
        n = x_sorted.size
        i = np.arange(1, n + 1)
        numerator = np.sum((2 * i - n - 1) * x_sorted)
        denominator = n * np.sum(x_sorted)
        
        return float(numerator / denominator)
    
    def get_info(self) -> Dict[str, Any]:
        """Return current step stats for logging"""
        return {
            # Bonuses (Positive)
            "reward/p2p_bonus_mean": np.mean(self.step_p2p_bonus),
            "reward/healthy_soc_bonus_mean": np.mean(self.step_healthy_soc_bonus),
            "reward/demand_satisfaction_bonus_mean": np.mean(self.step_demand_satisfaction_bonus),
            
            # Penalties (Negative)
            "reward/grid_import_penalty_mean": np.mean(self.step_grid_import_penalty),
            "reward/smoothing_penalty_mean": np.mean(self.step_smoothing_penalty),
            "reward/battery_cost_mean": np.mean(self.step_battery_cost),
            "reward/soc_penalty_mean": np.mean(self.step_soc_penalty),
            "reward/unmet_demand_penalty_mean": np.mean(self.step_unmet_demand_penalty),
            "reward/grid_overload_penalty_mean": np.mean(self.step_grid_overload_penalty),
            "reward/fairness_penalty": self.step_fairness_penalty,
            
            # Profit
            "reward/p2p_profit_mean": np.mean(self.step_p2p_profit),
            "reward/grid_cost_mean": np.mean(self.step_grid_cost),
            
            # Aggregates
            "reward/total_bonuses_mean": np.mean(
                self.step_p2p_bonus + 
                self.step_healthy_soc_bonus + 
                self.step_demand_satisfaction_bonus
            ),
            "reward/total_penalties_mean": np.mean(
                self.step_grid_import_penalty +
                self.step_smoothing_penalty +
                self.step_battery_cost +
                self.step_soc_penalty +
                self.step_unmet_demand_penalty +
                self.step_grid_overload_penalty
            )
        }
