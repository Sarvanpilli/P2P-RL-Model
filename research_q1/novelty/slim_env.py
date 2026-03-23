
import os
import sys
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import base class and components
from train.energy_env_recovery import EnergyMarketEnvRecovery
from research_q1.novelty.safety_layer import SafetyFilter
from research_q1.novelty.market_mechanism import DynamicMarketMechanism

# Safe import for MicrogridNode
try:
    from simulation.microgrid import MicrogridNode
except ImportError:
    from train.energy_env_recovery import MicrogridNode # Fallback if not direct

class EnergyMarketEnvSLIM(EnergyMarketEnvRecovery):
    """
    Safety-Constrained Liquidity-Integrated Market (SLIM) Environment.
    
    Novelty:
    1. Explicit Safety Layer (Projection) applied BEFORE physics.
    2. Liquidity Pool Mechanism for P2P clearing.
    3. Reward Reshaping (Feasibility + Profit).
    4. SCALABLE: Supports arbitrary N agents by cycling profiles.
    """

    def __init__(self, 
                 enable_safety: bool = True,
                 enable_p2p: bool = True,
                 market_type: str = "dynamic",
                 alpha_p2p: float = 0.01,
                 base_delta: float = 0.03,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.enable_safety = enable_safety
        self.enable_p2p = enable_p2p
        self.market_type = market_type
        self.alpha_p2p = alpha_p2p
        self.base_delta = base_delta
        
        # ── Fix 1: P2P completion bonus weight ($/kWh for any cleared P2P trade) ──
        self.p2p_bonus_per_unit = 0.15  # default; overridden by CurriculumCallback
        # ── Fix 2: Role-diversity penalty weight ($ per idle seller/buyer) ────────
        self.no_buyer_penalty_weight = 0.10
        # Grid penalty weight for beta_grid term
        self.grid_penalty_weight = 0.01
        
        # Initialize Safety Filters (always create, conditionally apply)
        self.safety_filters = []
        for node in self.nodes:
            sf = SafetyFilter(
                battery_capacity_kwh=node.battery_capacity_kwh,
                max_power_kw=node.battery_max_charge_kw, 
                efficiency=node.battery_eff
            )
            self.safety_filters.append(sf)
            
        # Initialize Market Mechanism
        if self.market_type == "dynamic":
            self.liquidity_pool = DynamicMarketMechanism(base_delta=self.base_delta)
        else:
            # Old naive auction mechanism (mid-price calculation)
            self.liquidity_pool = DynamicMarketMechanism(base_delta=0.0)
            
        # Logic: Base (7) + Market (4) = 11 base features + 2 weather + 4 type + (2 * H) forecast + 1 peer
        n_base = 7
        n_market = 4
        n_total_base = n_base + n_market
        n_weather = 2
        n_type = 4
        n_forecast = 2 * self.forecast_horizon
        n_peer = 1
        # Fix 3: +1 for the shared market_balance scalar appended at end of obs
        n_market_balance = 1
        
        obs_features = n_total_base + n_weather + n_type + n_forecast + n_peer
        # Total obs = per-agent features * n_agents + 1 shared market_balance signal
        self.obs_features_per_agent = obs_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents * obs_features + n_market_balance,),
            dtype=np.float32
        )

        # Tracking
        self.safety_violations = 0
        self.accumulated_profit = 0.0

    def set_reward_weights(self, p2p_bonus: float = 0.15,
                           no_buyer_penalty: float = 0.10,
                           grid_penalty: float = 0.01):
        """
        Dynamically update reward shaping weights.
        Called by CurriculumCallback via env_method() at training milestones.

        Args:
            p2p_bonus        $/kWh bonus for any cleared P2P trade (Fix 1)
            no_buyer_penalty $/idle-seller penalty when market has no buyers (Fix 2)
            grid_penalty     coefficient for the raw grid-import penalty in shaped reward
        """
        self.p2p_bonus_per_unit = float(p2p_bonus)
        self.no_buyer_penalty_weight = float(no_buyer_penalty)
        self.grid_penalty_weight = float(grid_penalty)

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.safety_violations = 0
        self.accumulated_profit = 0.0
        self.liquidity_pool.last_p2p_volume = 0.0
        self.liquidity_pool.last_delta = 0.03
        self.liquidity_pool.last_prices = {"seller_price": 0.08, "buyer_price": 0.20}
        return out


    def _setup_agents(self):
        """
        Override to support arbitrary N agents by cycling templates.
        Templates:
        0: Solar (5/10 kWh, 2.5/5 kW)
        1: Wind (5 kWh, 2.5 kW)
        2: EV (62 kWh, 7 kW)
        3: Standard (10 kWh, 5 kW)
        """
        self.nodes = []
        for i in range(self.n_agents):
            template_idx = i % 4
            
            if template_idx == 0: # Solar
                kwh = 5.0 if self.diversity_mode else 10.0
                kw = 2.5 if self.diversity_mode else 5.0
            elif template_idx == 1: # Wind
                kwh = 5.0
                kw = 2.5
            elif template_idx == 2: # EV
                kwh = 62.0
                kw = 7.0
            else: # Standard
                kwh = 10.0
                kw = 5.0
                
            node = MicrogridNode(
                node_id=i,
                battery_capacity_kwh=kwh,
                battery_max_charge_kw=kw,
                battery_eff=0.9
            )
            if self.enable_ramp_rates:
                node.max_ramp_kw = kw
                
            self.nodes.append(node)

    def _get_current_data(self):
        """
        Override to map N agents to 4 data columns (cycling).
        """
        row = self.df.iloc[self.current_idx]
        
        demands = []
        generations = []
        
        for i in range(self.n_agents):
            template_idx = i % 4
            # Map to base columns
            d_col = f'agent_{template_idx}_demand'
            g_col = f'agent_{template_idx}_pv' if template_idx != 1 else f'agent_{template_idx}_wind'
            
            demands.append(row.get(d_col, 0.0))
            generations.append(row.get(g_col, 0.0))
            
        # Weather is global
        temp = row.get('temperature_2m', 20.0)
        wind = row.get('windspeed_100m', 5.0)
        
        return np.array(demands), np.array(generations), np.array([temp, wind])

    def _apply_ev_constraints(self):
        """
        Override to apply constraints to ALL EV agents (idx % 4 == 2).
        """
        hour = (self.current_idx % 24)
        is_away = (8 <= hour < 17)
        
        for i in range(self.n_agents):
            if (i % 4) == 2: # EV Agent
                ev_node = self.nodes[i]
                if is_away:
                     ev_node.battery_capacity_kwh = 2.0
                     ev_node.soc = min(ev_node.soc, ev_node.battery_capacity_kwh)
                else:
                     ev_node.battery_capacity_kwh = 62.0


    def _get_obs(self, total_export, total_import):
        """
        Override to support arbitrary N agents (Scalable).

        Fix 3: Appends a single shared `market_balance` scalar at the end of
        the flattened observation vector.
          market_balance = (total_pv_generation - total_demand) / (total_demand + 1e-6)
          Positive  → community surplus  (market needs buyers)
          Negative  → community deficit  (market needs sellers)
        This gives the GNN a direct signal to self-organise complementary roles.
        obs_dim: n_agents * obs_features_per_agent + 1  (104 + 1 = 105 for N=4, H=4)
        """
        # Calculate derived features
        hour = (self.current_idx % 24)
        sin_time = np.sin(2 * np.pi * hour / 24)
        cos_time = np.cos(2 * np.pi * hour / 24)
        
        buy, sell = self._get_grid_prices()
        
        # Get Data (Uses our overridden scalable method)
        demands, generations, weather = self._get_current_data()
        
        # Calculate Peer Demand
        current_net_loads = generations - demands 
        
        # Fix 3: Community-level market balance signal
        total_generation = np.sum(generations)
        total_demand_sum = np.sum(demands)
        market_balance = (total_generation - total_demand_sum) / (total_demand_sum + 1e-6)
        market_balance = float(np.clip(market_balance, -3.0, 3.0))  # bounded
        
        # Normalize features roughly
        weather_norm = weather / [40.0, 20.0] 
        
        obs_list = []
        for i in range(self.n_agents):
            node = self.nodes[i]
            template_idx = i % 4 # Cyclical mapping
            
            # Peer Demand
            peer_surplus = np.sum(current_net_loads) - current_net_loads[i]
            peer_demand = -peer_surplus 
            peer_demand /= 10.0 
            
            # Forecasts (Scalable lookup)
            forecasts = []
            for h in range(1, self.forecast_horizon + 1):
                idx = min(self.current_idx + h, self.max_idx)
                row = self.df.iloc[idx]
                
                # Map based on template
                d_col = f'agent_{template_idx}_demand'
                g_col = f'agent_{template_idx}_pv' if template_idx != 1 else f'agent_{template_idx}_wind'
                
                d = row.get(d_col, 0.0)
                g = row.get(g_col, 0.0)
                
                forecasts.extend([d/5.0, g/5.0]) 
            
            # Agent Type One-Hot (Fixed size 4, mapped by template)
            type_vec = np.zeros(4)
            type_vec[template_idx] = 1.0 # Corrected from [i] to [template_idx]
            
            base = [
                node.soc / 62.0, 
                buy,
                sell,
                sin_time,
                cos_time,
                total_export / 20.0,
                total_import / 20.0,
                self.liquidity_pool.last_p2p_volume / 20.0,
                self.liquidity_pool.last_delta / 0.1,
                self.liquidity_pool.last_prices["seller_price"] / 0.5,
                self.liquidity_pool.last_prices["buyer_price"] / 0.5
            ]
            
            # Concatenate per-agent features
            full_obs = np.concatenate([
                base,
                weather_norm,
                type_vec,
                forecasts,
                [peer_demand]
            ])
            
            obs_list.extend(full_obs)
        
        # Fix 3: Append shared market_balance at the very end
        obs_list.append(market_balance)
        
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        """
        Override step to inject Safety and Market layers.
        """

        # 0. Formatting
        raw_action = action.reshape(self.n_agents, 2)
        
        # === PRE-STEP: Update Dynamic Constraints ===
        # CRITICAL: Must apply EV constraints BEFORE Safety Projection
        self._apply_ev_constraints()
        
        # Arrays to hold modified actions
        safe_battery_actions = np.zeros(self.n_agents)
        p2p_requests = np.zeros(self.n_agents)
        
        # === LAYER 2: Safety Filter ===
        total_feasibility_penalty = 0.0
        
        for i in range(self.n_agents):
            node = self.nodes[i]
            sf = self.safety_filters[i]
            
            # Action 0: Battery (Normalized [-1, 1] -> Scale to Max kW)
            act_batt_norm = raw_action[i, 0]
            req_batt_kw = act_batt_norm * node.battery_max_charge_kw
            
            # Action 1: P2P (Normalized [-1, 1] -> Scale to Max kW)
            act_p2p_norm = raw_action[i, 1]
            req_p2p_kw = act_p2p_norm * 5.0 # Max P2P rate (Fixed for now or scale?)
            

            # Project logic (Always run for tracking)
            current_cap = node.battery_capacity_kwh
            
            feasible_batt_kw = sf.project_action(
                current_soc=node.soc,
                desired_action_kw=req_batt_kw,
                dt_hours=self.timestep_hours,
                current_capacity=current_cap 
            )
            



            # Check for violation (Tracking)
            violation_mag = abs(feasible_batt_kw - req_batt_kw)
            is_violation = violation_mag > 0.01
            
            if is_violation:
                self.safety_violations += 1
            
            if self.enable_safety:
                # Apply Safety Layer
                safe_batt_kw = feasible_batt_kw
                
                # Apply Penalty
                if is_violation:
                    total_feasibility_penalty -= 0.1 * violation_mag
            else:
                # No Safety: Ignore projection, pass raw request
                # (Physics will clip it later, but we tracked the 'intent' to violate)
                safe_batt_kw = req_batt_kw

            safe_battery_actions[i] = safe_batt_kw
            p2p_requests[i] = req_p2p_kw
            
        # === LAYER 3: Liquidity Matching ===

        # Update Market Prices
        retail, feed_in = self._get_grid_prices()
        self.liquidity_pool.update_prices(retail, feed_in)
        
        if self.enable_p2p:
            # Clear Market using Dynamic Mechanism. Invert requests to match Market's convention (negative=sell)
            market_output = self.liquidity_pool.clear_market(-p2p_requests)
            
            # Invert trades back to match Environment's convention (positive=sell)
            cleared_p2p = -np.array(market_output["p2p_trades"])
            
            seller_revenue = np.array(market_output["seller_revenue"])
            buyer_cost = np.array(market_output["buyer_cost"])
            grid_import_arr = np.array(market_output["grid_import"])
            grid_export_arr = np.array(market_output["grid_export"])
            p2p_vol = market_output["p2p_volume"]
            seller_price = market_output["seller_price"]
            buyer_price = market_output["buyer_price"]
            delta = market_output["delta"]
        else:
            # No P2P: Force zero P2P trades, all residual goes to grid
            cleared_p2p = np.zeros(self.n_agents)
            buyer_cost = np.zeros(self.n_agents)
            seller_revenue = np.zeros(self.n_agents)
            grid_import_arr = np.maximum(p2p_requests, 0)
            grid_export_arr = np.abs(np.minimum(p2p_requests, 0))
            p2p_vol = 0.0
            seller_price = feed_in
            buyer_price = retail
            delta = 0.0

        
        # === PHYSICS EXECUTION ===
        rewards = np.zeros(self.n_agents)
        demands, generations, weather = self._get_current_data()
        
        total_export = 0.0
        total_import = 0.0
        
        for i in range(self.n_agents):
            node = self.nodes[i]
            
            # Execute Safe Battery Action
            node_res = node.step(
                battery_action_kw=safe_battery_actions[i],
                current_demand_kw=demands[i],
                current_pv_kw=generations[i],
                dt_hours=self.timestep_hours
            )
            
            physical_net_load = node_res['net_load_kw']
            
            # Grid Flow = Physical Need + P2P Action
            # Seller: net_load(-5) + cleared(+3) = grid_flow(-2) -> 2kW export to grid
            # Buyer: net_load(+5) + cleared(-2) = grid_flow(+3) -> 3kW import from grid
            # Cheater: net_load(-1) + cleared(+5) = grid_flow(+4) -> 4kW expensive import penalty!
            grid_flow = physical_net_load + cleared_p2p[i]
            
            # The REAL grid import/export must be strictly defined by final physical flow
            # The market mechanism's 'unmatched' intent is an illusion if the physics can't deliver it.
            actual_grid_import = max(0.0, grid_flow)
            actual_grid_export = abs(min(0.0, grid_flow))
            
            # Replace the intent-based grid variables with ground-truth physics
            grid_import_arr[i] = actual_grid_import
            grid_export_arr[i] = actual_grid_export
            
            # 1. P2P Settlement (based on cleared trades)
            p2p_profit = seller_revenue[i] - buyer_cost[i]
            
            # 2. Grid Settlement (based on final physical flow)
            grid_cost = actual_grid_import * retail
            grid_rev = actual_grid_export * feed_in
            
            total_import += actual_grid_import
            total_export += actual_grid_export
            
            # Base financial profit
            financial_profit = p2p_profit + grid_rev - grid_cost
            
            # Explicit Reward Shaping (Incentivize P2P, gentle grid penalty)
            # grid_penalty_weight is adjustable by CurriculumCallback
            agent_p2p_vol = abs(cleared_p2p[i])
            shaped_reward = (financial_profit
                             + (self.alpha_p2p * agent_p2p_vol)
                             - (self.grid_penalty_weight * actual_grid_import))
            
            # ── Fix 1: P2P Completion Bonus (both buyer and seller) ──────────────
            # Rewarding the act of clearing a trade (not just the financial outcome)
            # makes BUYING explicitly rewarding, breaking the all-seller equilibrium.
            if agent_p2p_vol > 0.01:
                p2p_completion_bonus = agent_p2p_vol * self.p2p_bonus_per_unit
                shaped_reward += p2p_completion_bonus
            
            rewards[i] = shaped_reward
            self.accumulated_profit += financial_profit
        
        # ── Fix 2: Role Diversity Penalty ────────────────────────────────────────
        # Penalise timesteps where the market has only sellers or only buyers,
        # because a one-sided market clears zero P2P trades.
        n_sellers = int(np.sum(cleared_p2p > 0.01))
        n_buyers  = int(np.sum(cleared_p2p < -0.01))
        diversity_penalty = 0.0

        if n_sellers > 0 and n_buyers == 0:
            # All agents are selling – nobody to trade with
            diversity_penalty = self.no_buyer_penalty_weight * n_sellers
            rewards -= diversity_penalty / self.n_agents  # spread penalty over agents
        elif n_buyers > 0 and n_sellers == 0:
            # All agents are buying – nobody to supply them
            diversity_penalty = self.no_buyer_penalty_weight * n_buyers
            rewards -= diversity_penalty / self.n_agents

        total_reward_scalar = float(np.sum(rewards))

        # Global State Updates
        self.current_idx += 1
        self.timestep_count += 1
        self.p2p_trades_count += 1 if np.sum(np.abs(cleared_p2p)) > 0 else 0
        
        # Check termination
        done = False
        truncated = False
        if self.current_idx >= self.max_idx:
            truncated = True
        if self.timestep_count >= self.max_steps:
            truncated = True
            
        # Observation
        obs = self._get_obs(total_export, total_import)
        
        # Per-agent bonus tracking for TensorBoard
        step_p2p_bonus = sum(
            abs(cleared_p2p[i]) * self.p2p_bonus_per_unit
            for i in range(self.n_agents) if abs(cleared_p2p[i]) > 0.01
        )

        info = {
            'p2p_volume': p2p_vol,
            'grid_import': total_import,
            'grid_export': total_export,
            'seller_price': seller_price,
            'buyer_price': buyer_price,
            'delta': delta,
            'profit': self.accumulated_profit,
            'safety_violations': self.safety_violations,
            # ── Metrics for TensorBoard ──────────────────────────────────────────
            'reward/p2p_bonus': step_p2p_bonus,
            'reward/no_buyer_penalty': diversity_penalty,
            'market/n_buyers': n_buyers,
            'market/n_sellers': n_sellers,
            'market/p2p_volume': p2p_vol,
        }
        
        return obs, total_reward_scalar, done, truncated, info

