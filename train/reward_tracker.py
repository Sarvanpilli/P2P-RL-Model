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
        self.step_smoothing_penalty = np.zeros(self.n_agents)
        self.step_co2_penalty = np.zeros(self.n_agents)
        self.step_p2p_bonus = np.zeros(self.n_agents)
        self.step_alignment_reward = np.zeros(self.n_agents)
        self.step_participation_penalty = np.zeros(self.n_agents)
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

    """
    Economic intuition:
    Grid penalty is capped at 0.15 to stay below the P2P profit margin of $0.10–0.40/kWh.
    If penalty > profit margin, agents will prefer grid over P2P.
    """
    def calculate_total_reward(self, 
                             profits: np.ndarray, 
                             grid_import_penalties: np.ndarray,
                             soc_penalties: np.ndarray,
                             grid_overload_costs: np.ndarray,
                             battery_costs: np.ndarray,
                             smoothing_penalties: np.ndarray = None,
                             co2_penalties: np.ndarray = None,
                             p2p_bonuses: np.ndarray = None,
                             total_export_kw: float = 0.0,
                             traded_energy: np.ndarray = None,
                             trade_intent: np.ndarray = None,
                             alpha: float = 0.05,
                             beta: float = 1.0,
                             lambda_align: float = 0.0,
                             avg_market_price: float = 0.15,
                             bids: np.ndarray = None,
                             use_alignment_reward: bool = True) -> float:
        """
        Aggregates the core components into a scalar reward.
        Includes Phase 5+ upgrades: Trade Success and Diversity Regularization.
        
        Args:
            alpha: Dynamic reward coefficient for trade success (Step 3 Curriculum).
        """
        
        # 1. Update internal state with Clipping
        # ... (rest of logic)
        self.step_profit = profits
        self.step_grid_penalty = np.clip(grid_import_penalties, 0.0, 50.0) 
        self.step_soc_penalty = np.clip(soc_penalties, 0.0, 10.0)
        self.step_grid_overload_penalty = np.clip(grid_overload_costs, 0.0, 10.0)
        self.step_battery_cost = np.clip(battery_costs, 0.0, 50.0)
        
        # Initialize optional components if None
        if smoothing_penalties is None: smoothing_penalties = np.zeros(self.n_agents)
        if co2_penalties is None: co2_penalties = np.zeros(self.n_agents)
        if p2p_bonuses is None: p2p_bonuses = np.zeros(self.n_agents)
        if traded_energy is None: traded_energy = np.zeros(self.n_agents)
        
        self.step_smoothing_penalty = np.clip(smoothing_penalties, 0.0, 10.0)
        self.step_co2_penalty = np.clip(co2_penalties, 0.0, 20.0)
        self.step_p2p_bonus = p2p_bonuses
        
        # 3g. Trade Success Reward (alpha * traded_energy)
        self.step_trade_reward = traded_energy * alpha
        
        # --- PHASE 13: SCIENTIFIC VALIDATION ABLATIONS ---
        if use_alignment_reward and bids is not None:
             # bids = n_agents x 4 (qty, price, agent_id, role)
             # Extract prices for coordination
             agent_prices = bids[:, 1]
             # exp(-abs(bid - avg))
             alignment_signals = np.exp(-np.abs(agent_prices - avg_market_price))
             self.step_alignment_reward = alignment_signals * lambda_align
        else:
             self.step_alignment_reward = np.zeros(self.n_agents)

        # 3i. Participation Penalty
        if trade_intent is not None:
             self.step_participation_penalty = np.where(np.abs(trade_intent) < 0.05, -0.01, 0.0)
        else:
             self.step_participation_penalty = np.zeros(self.n_agents)
             
        # grid_penalty now uses BETA
        self.step_grid_penalty = np.clip(grid_import_penalties * beta, 0.0, 50.0)
        
        # 2. Compute per-agent net value (before global penalties)
        per_agent_net = (self.step_profit 
                         + self.step_p2p_bonus
                         + self.step_trade_reward
                         + self.step_alignment_reward
                         + self.step_participation_penalty
                         - self.step_grid_penalty 
                         - self.step_soc_penalty 
                         - self.step_grid_overload_penalty 
                         - self.step_battery_cost
                         - self.step_smoothing_penalty
                         - self.step_co2_penalty)
        
        # 3. Compute Fairness Penalty (Global)
        # Reduced by 0.5 as requested to avoid over-regularization
        gini = self.compute_gini(per_agent_net)
        self.step_fairness_penalty = 0.5 * self.fairness_coeff * gini * (1.0 + 0.05 * abs(total_export_kw))
        
        # 4. Diversity Regularization (gamma * var / (var + 1))
        # Encourages agents to take different roles (buying vs selling)
        GAMMA = 0.1
        if trade_intent is not None:
             var_intent = np.var(trade_intent)
             self.step_diversity_reward = GAMMA * (var_intent / (var_intent + 1.0))
        else:
             self.step_diversity_reward = 0.0
             
        # 5. Total Shared Reward
        total_reward = np.sum(per_agent_net) - self.step_fairness_penalty + self.step_diversity_reward
        
        return float(total_reward)

    def get_profit_breakdown(self) -> Dict[str, Any]:
        """
        Returns a financial breakdown of the current step for all agents.
        """
        gross_profit = np.sum(self.step_profit)
        total_bonuses = np.sum(self.step_p2p_bonus)
        
        # Sum of all penalty terms
        penalty_terms = (
            np.sum(self.step_grid_penalty) +
            np.sum(self.step_soc_penalty) +
            np.sum(self.step_grid_overload_penalty) +
            np.sum(self.step_battery_cost) +
            np.sum(self.step_smoothing_penalty) +
            np.sum(self.step_co2_penalty) +
            self.step_fairness_penalty
        )
        
        # net_reward = (gross_profit + total_bonuses) - total_penalties
        net_reward = (gross_profit + total_bonuses) - penalty_terms
        
        # P2P Volume calculation: p2p_bonus = volume * 0.20
        # So p2p_volume = total_bonuses / 0.20
        p2p_volume = total_bonuses / 0.20 if total_bonuses > 0 else 0.0
        
        return {
            'gross_profit': float(gross_profit),
            'total_penalties': float(penalty_terms),
            'net_reward': float(net_reward),
            'p2p_volume_kwh': float(p2p_volume),
            'is_profitable': bool(net_reward > 0)
        }



    def calculate_scientific_metrics(self, battery_degradation_rate: float = 0.01, co2_intensity: float = 0.4):
        """
        Computes scientific metrics for Phase 13 validation.
        """
        clean_profits = self.step_profit
        # Note: step_battery_quantities is what we want for degradation calculation
        # If it's not explicitly tracked, we use step_battery_cost as a proxy or track it in env.
        # For Phase 13, let's assume battery_costs already includes some factor, 
        # but the request asks for a separate metric.
        
        # We need to ensure step_grid_import_quantities exists
        grid_import_q = getattr(self, "step_grid_import_quantities", np.zeros(self.n_agents))
        battery_q = getattr(self, "step_battery_quantities", np.zeros(self.n_agents))
        
        economic_profits = clean_profits - (battery_q * battery_degradation_rate)
        carbon_emissions = grid_import_q * co2_intensity
        
        return {
            "clean_profits": clean_profits,
            "economic_profits": economic_profits,
            "carbon_emissions": carbon_emissions
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns dict of current step stats for logging."""
        return {
            "mean_profit": np.mean(self.step_profit),
            "mean_grid_penalty": np.mean(self.step_grid_penalty),
            "mean_soc_penalty": np.mean(self.step_soc_penalty),
            "mean_grid_overload_penalty": np.mean(self.step_grid_overload_penalty),
            "mean_battery_cost": np.mean(self.step_battery_cost),
            "mean_smoothing_penalty": np.mean(self.step_smoothing_penalty),
            "mean_co2_penalty": np.mean(self.step_co2_penalty),
            "mean_p2p_bonus": np.mean(self.step_p2p_bonus),
            "mean_alignment_reward": np.mean(self.step_alignment_reward),
            "mean_participation_penalty": np.mean(self.step_participation_penalty),
            "fairness_penalty": self.step_fairness_penalty
        }
