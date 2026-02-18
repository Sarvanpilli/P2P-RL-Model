
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
from research_q1.novelty.market_mechanism import LiquidityPool

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
                 **kwargs):
        super().__init__(**kwargs)
        
        self.enable_safety = enable_safety
        self.enable_p2p = enable_p2p
        
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
        self.liquidity_pool = LiquidityPool()
        

        # Tracking
        self.safety_violations = 0
        self.accumulated_profit = 0.0

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.safety_violations = 0
        self.accumulated_profit = 0.0
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
                total_import / 20.0
            ]
            
            # Concatenate
            full_obs = np.concatenate([
                base,
                weather_norm,
                type_vec,
                forecasts,
                [peer_demand]
            ])
            
            obs_list.extend(full_obs)
        
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
            # Clear Market
            cleared_p2p, clearing_price, market_stats = self.liquidity_pool.clear_market(p2p_requests)
        else:
            # No P2P: Force zero P2P trades
            cleared_p2p = np.zeros(self.n_agents)
            clearing_price = (retail + feed_in) / 2.0
            market_stats = {'traded_volume': 0.0}

        
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
            
            # Grid Flow = Physical Need - P2P Support
            grid_flow = physical_net_load - cleared_p2p[i]
            
            # Financial Calculation
            step_reward = 0.0
            
            # 1. P2P Settlement
            p2p_cost = cleared_p2p[i] * clearing_price
            step_reward -= p2p_cost
            
            # 2. Grid Settlement
            if grid_flow > 0: # Import
                grid_cost = grid_flow * retail
                step_reward -= grid_cost
                total_import += grid_flow
            else: # Export
                grid_rev = abs(grid_flow) * feed_in
                step_reward += grid_rev
                total_export += abs(grid_flow)
                
            rewards[i] = step_reward
            self.accumulated_profit += step_reward
        
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
        
        info = {
            'clearing_price': clearing_price,
            'p2p_volume': market_stats['traded_volume'],
            'profit': np.sum(rewards)
        }
        
        return obs, np.sum(rewards), done, truncated, info

