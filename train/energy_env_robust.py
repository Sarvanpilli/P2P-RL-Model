
"""
EnergyMarketEnvRobust: Research-Grade Gymnasium Environment for P2P Energy Trading.
Refactored to use MicrogridNode and Real Data.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional

# Local modules
try:
    from train.autonomous_guard import AutonomousGuard
    from market.matching_engine import MatchingEngine
    from train.reward_tracker import RewardTracker
    from simulation.microgrid import MicrogridNode
except ImportError:
    from .autonomous_guard import AutonomousGuard
    from ..market.matching_engine import MatchingEngine
    from .reward_tracker import RewardTracker
    from ..simulation.microgrid import MicrogridNode

class EnergyMarketEnvRobust(gym.Env):
    """
    Consolidated Environment using MicrogridNode physics and Real Data.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, 
                 n_prosumers=4, 
                 n_consumers=0, 
                 data_file="processed_hybrid_data.csv", 
                 random_start_day=True,
                 enable_ramp_rates=True,
                 enable_losses=True,
                 forecast_horizon=4, 
                 enable_predictive_obs=True,
                 forecast_noise_std=0.0,
                 use_alignment_reward=True,   # PHASE 13
                 use_curriculum=True):        # PHASE 13
        
        super().__init__()
        
        self.n_prosumers = n_prosumers
        self.n_consumers = n_consumers
        self.n_agents = n_prosumers + n_consumers
        self.data_file = data_file
        self.random_start_day = random_start_day
        self.enable_ramp_rates = enable_ramp_rates
        self.enable_losses = enable_losses
        self.forecast_horizon = forecast_horizon
        self.enable_predictive_obs = enable_predictive_obs
        self.forecast_noise_std = forecast_noise_std
        self.use_alignment_reward = use_alignment_reward
        self.use_curriculum = use_curriculum
        self.beta = 1.0 
        
        # Physics Constants
        self.max_line_capacity_kw = 50.0
        self.line_resistance_ohms = 0.05
        self.overload_multiplier = 5.0 

        self.rng = np.random.default_rng(seed=42) 
        self._load_data()
        
        # Setup Agents
        self._setup_agents()
        
        # History & Market
        self.history_window_size = 4
        self.history_buffer = {
            'demand': np.zeros((self.n_agents, 4)),
            'pv': np.zeros((self.n_agents, 4))
        }
        
        from collections import deque
        self.market_history = {
            'prices': deque(maxlen=24),
            'success_flags': deque(maxlen=24),
            'last_market_price': 0.15,
            'last_success_rate': 0.0,
            'steps_without_trade': 0
        }
        
        self.prev_actions = np.zeros((self.n_agents, 3))
        
        # Cumulative metrics for evaluation
        self.total_grid_import = 0.0
        self.total_grid_export = 0.0
        self.total_p2p_volume = 0.0
        self.total_clean_profit = 0.0      # PHASE 13
        self.total_economic_profit = 0.0   # PHASE 13
        self.total_carbon_emissions = 0.0  # PHASE 13
        self.total_demand_all = 1e-6
        self.rolling_grid_dep_buffer = deque([1.0]*24, maxlen=24)
        self.rolling_grid_dependency = 1.0

        self.matching_engine = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)
        self.timestep_hours = 1.0
        self.grid_voltage_kv = 0.4
        
        # --- PHASE 12: COORDINATION CURRICULUM ---
        self.total_training_steps = 0  
        self.curriculum_decay_period = 150000 
        
        # Step 4: Matching Engine schedules (Used only if use_curriculum=True)
        self.curr_eps_start = 0.20
        self.curr_eps_final = 0.10
        self.curr_margin_start = 0.05
        self.curr_margin_final = 0.0
        
        # Step 3: Reward Schedules
        self.curr_alpha_start = 0.10
        self.curr_alpha_final = 0.03
        self.curr_beta_start = 3.0
        self.curr_beta_final = 2.0
        self.curr_lambda_align_start = 0.05
        self.curr_lambda_align_final = 0.0
        
        # Step 4: Action Noise
        self.curr_noise_std_start = 0.25
        self.curr_noise_std_final = 0.0
        
        agent_specs = [{"capacity": n.battery_capacity_kwh, "max_charge": n.battery_max_charge_kw} for n in self.nodes]
        self.guard = AutonomousGuard(
            n_agents=self.n_agents,
            agent_specs=agent_specs,
            timestep_hours=self.timestep_hours
        )
        
        self.reward_tracker = RewardTracker(self.n_agents)
        
        # Lagrangian safety layer
        from train.lagrangian_safety import LagrangianSafetyLayer
        self.lagrangian = LagrangianSafetyLayer(
            n_agents=self.n_agents,
            alpha=0.005,
            threshold_soc_violation=0.01,
            threshold_line_violation=0.05,
            threshold_voltage_violation=0.03,
            max_lambda=10.0,
            lambda_init=0.1,
        )
        self.max_line_capacity_kw = getattr(self, 'max_line_capacity_kw', 50.0)

        
        # Spaces
        # heterogeneous physical bounds for each agent
        lows = []
        highs = []
        for node in self.nodes:
            lows.extend([-node.battery_max_charge_kw, -self.max_line_capacity_kw, 0.0])
            highs.extend([node.battery_max_charge_kw, self.max_line_capacity_kw, 1.0])
        
        self.action_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32)
        )
        
        # Observation: 
        # Base: SOC, Retail, FeedIn, SinTime, CosTime, Exp, Imp (7)
        # + History (Dem, PV) ? Maybe not requested explicitly, but good.
        # + Forecast (Dem, PV, Wind) x Horizon
        # + Static One-Hot (4 types)
        # + Weather (Temp, WindSpeed)
        
        # Calculate Obs Dim
        # Base (7)
        # + Weather (2)
        # + One-Hot Agent Type (4)
        # + Forecast: 4 hours * (Solar + Wind + Demand) ?? 
        #   Agent 1 sees Wind forecast, Agent 0 sees Solar. 
        #   Let's give relevant forecast.
        #   Forecast features: [Dem_t+k, Gen_t+k] * H
        #   Where Gen is PV for Solar agent, Wind for Wind agent.
        #   So 2 * H features.
        # Calculate Obs Dim
        # Base (7)
        # + Weather (2)
        # + One-Hot Agent Type (4)
        # + Forecast: 4 hours * (Solar + Wind + Demand) -> 2 features * H
        # History (2) was planned but not implemented in _get_obs. Removing.
        
        # Updated Obs Dim: 4 local + 4 context + 4 global + (2 * self.forecast_horizon)
        # Total: 12 + 8 = 20 per agent (for H=4)
        self.n_obs_features = 4 + 4 + 4 + (2 * self.forecast_horizon)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_obs_features,), 
            dtype=np.float32
        )
        
        # Action Space: [Battery, Trade Amount, Price Bid]
        self.action_space = spaces.Box(
             low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        self.current_idx = 0
        self.day_start_idx = 0
        self.max_steps = 24 * 7 

        # Market State History for observations
        from collections import deque
        self.market_history = {
            'prices': deque(maxlen=24),
            'success_flags': deque(maxlen=24),
            'last_market_price': 0.15,
            'last_success_rate': 0.0,
            'steps_without_trade': 0
        }

        # Track prev action for smoothing
        self.prev_actions = np.zeros((self.n_agents, 3))
        
        # Cumulative metrics for evaluation
        self.total_grid_import = 0.0
        self.total_grid_export = 0.0
        self.total_p2p_volume = 0.0
        self.total_demand_all = 1e-6
        
        # --- NEW: Rolling Dependency Tracker (24-step) ---
        self.rolling_grid_dependency = 1.0

        self.prev_market_price = 0.15 # Step 2/4 Persistence
        self.rng = np.random.default_rng(seed=42)
        
    def _setup_agents(self):
        """Rule-based proportional agent distribution (Step 1)."""
        self.nodes = []
        indices = np.arange(self.n_agents)
        self.rng.shuffle(indices)
        
        # Calculate counts
        n_hi = int(0.25 * self.n_agents)
        n_vhi = max(1, int(0.1 * self.n_agents))
        n_heavy = int(0.25 * self.n_agents)
        
        role_map = {}
        for i, idx in enumerate(indices):
            if i < n_hi:
                role_map[idx] = 'hi_solar'
            elif i < n_hi + n_vhi:
                role_map[idx] = 'vhi_solar'
            elif i < n_hi + n_vhi + n_heavy:
                role_map[idx] = 'heavy_consumer'
            else:
                # Remaining: Proprosed Prosumer/Consumer split
                if idx < self.n_prosumers:
                    role_map[idx] = 'prosumer'
                else:
                    role_map[idx] = 'consumer'
        
        for i in range(self.n_agents):
            role = role_map[i]
            
            # Base logic
            if role in ['hi_solar', 'vhi_solar', 'prosumer']:
                # Prosumer variants
                node = MicrogridNode(node_id=i, battery_capacity_kwh=7.5, battery_max_charge_kw=3.5, battery_eff=0.9)
                node.is_prosumer = 1.0
                node.dem_mult = 1.0
                if role == 'hi_solar':
                    node.gen_mult = 1.5
                    node.agent_type_id = 1
                elif role == 'vhi_solar':
                    node.gen_mult = 2.0
                    node.agent_type_id = 1
                else:
                    node.gen_mult = 1.0
                    node.agent_type_id = 0
            else:
                # Consumer variants
                node = MicrogridNode(node_id=i, battery_capacity_kwh=0.0, battery_max_charge_kw=0.0, battery_eff=0.9)
                node.is_prosumer = 0.0
                node.gen_mult = 0.0
                if role == 'heavy_consumer':
                    node.dem_mult = 2.0
                    node.agent_type_id = 4
                else:
                    node.dem_mult = 1.2
                    node.agent_type_id = 4
            
            # Set price sensitivity [0.8, 1.2] (Step 4)
            node.price_sensitivity = self.rng.uniform(0.8, 1.2)
            self.nodes.append(node)

    def _load_data(self):
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"{self.data_file} not found.")
        
        self.df = pd.read_csv(self.data_file)
        self.max_idx = len(self.df) - 24 # Safety buffer
        
        # Pre-fetch normalization max values if possible? 
        # Assume data is already normalized [0,1] as per user request in preprocessing.
        # "To help PPO converge, create ... version ... scaled to [0, 1]"
        # So we trust the data is [0, 1].
        pass

    def _apply_ev_constraints(self):
        """Phase 5: Dynamic logic for EV agents."""
        hour = (self.current_idx % 24)
        for node in self.nodes:
            if node.agent_type_id == 2:
                # Day: 08:00 - 17:00
                if 8 <= hour < 17:
                     node.battery_capacity_kwh = 2.0
                     if node.soc > 2.0:
                         node.soc = 2.0
                else:
                     # Night: 62kWh
                     if node.battery_capacity_kwh < 62.0:
                         node.battery_capacity_kwh = 62.0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        
        # Random start day
        if self.df is not None:
            max_start = len(self.df) - 25 # Assume minimum episode 24h
            if self.random_start_day and max_start > 0:
                self.day_start_idx = self.rng.integers(0, max_start)
            else:
                self.day_start_idx = 0
        else:
             self.day_start_idx = 0
        
        self.timestep_count = 0
        self.current_idx = self.day_start_idx
        
        # Reset Nodes
        for node in self.nodes:
            start_soc = self.rng.uniform(0.2, 0.8) * node.battery_capacity_kwh
            node.reset(soc=start_soc)
            
        # Reset History
        self.history_buffer['demand'].fill(0.0)
        self.history_buffer['pv'].fill(0.0)
        self.prev_actions.fill(0.0) # Unified Plural
        
        # Reset Cumulative Metrics
        self.total_grid_import = 0.0
        self.total_grid_export = 0.0
        self.total_p2p_volume = 0.0
        
        # Pre-fill history?
        # Ideally we loop back 4 steps. For now, zero padding is okay or we can read back.
        # Let's read back for correctness if possible.
        if self.df is not None:
             for k in range(self.history_window_size):
                 idx = self.current_idx - (self.history_window_size - k)
                 row = self.df.iloc[idx % len(self.df)]
                 for i in range(self.n_agents):
                    self.history_buffer['demand'][i, k] = row.get(f"agent_{i}_demand_kw", 0.0)
                    self.history_buffer['pv'][i, k] = row.get(f"agent_{i}_pv_kw", 0.0)
        
        # End-of-episode lambda update (must happen before state reset)
        if hasattr(self, 'lagrangian'):
            self.lagrangian.end_episode_update()
            
        self.reward_tracker.reset()
        self.guard.reset()
        
        return self._get_obs(0.0, 0.0), {}

    def _get_current_data(self):
        """
        Fetch data for current timestep from hybrid dataset.
        Returns: demands, generations, weather(temp, windspeed)
        """
        if self.df is None:
            # Fallback
            return np.zeros(self.n_agents), np.zeros(self.n_agents), np.zeros(2)
            
        row = self.df.iloc[self.current_idx]
        
        # Demand (All have demand)
        dems = []
        gens = []
        
        # Agent 0: Solar, Demand
        dems.append(row.get('agent_0_demand', row.get('agent_0_demand_kw', 0.0)))
        gens.append(row.get('agent_0_pv', row.get('agent_0_pv_kw', 0.0)))
        
        # Agent 1: Wind/Solar, Demand
        dems.append(row.get('agent_1_demand', row.get('agent_1_demand_kw', 0.0)))
        gens.append(row.get('agent_1_wind', row.get('agent_1_pv_kw', row.get('agent_1_pv', 0.0))))
        
        # Agent 2: EV, Demand
        dems.append(row.get('agent_2_demand', row.get('agent_2_demand_kw', 0.0)))
        gens.append(row.get('agent_2_pv', row.get('agent_2_pv_kw', 0.0)))
        
        # Agent 3: Standard
        dems.append(row.get('agent_3_demand', row.get('agent_3_demand_kw', 0.0)))
        gens.append(row.get('agent_3_pv', row.get('agent_3_pv_kw', 0.0)))
        
        # Apply Multipliers (Agent Diversity Step 4)
        scaled_dems = []
        scaled_gens = []
        # Support n_agents > 4 or use modulo if needed, but here we assume n_agents=4
        for i in range(self.n_agents):
            node = self.nodes[i]
            scaled_dems.append(dems[i % len(dems)] * getattr(node, 'dem_mult', 1.0))
            scaled_gens.append(gens[i % len(gens)] * getattr(node, 'gen_mult', 1.0))
            
        # Weather
        temp = row.get('temperature_2m', 20.0)
        wind = row.get('windspeed_100m', 5.0)
        weather = np.array([temp, wind])
        
        return np.array(scaled_dems), np.array(scaled_gens), weather
        
    def update_training_step(self, step: int):
        """Step 11 Core: Sync global progress from callback."""
        self.total_training_steps = step

    def _get_curriculum_params(self) -> Dict[str, float]:
        """Calculates Phase 12 dynamic curriculum parameters."""
        progress = min(1.0, self.total_training_steps / self.curriculum_decay_period)
        
        # Linear decays
        epsilon = self.curr_eps_start - (self.curr_eps_start - self.curr_eps_final) * progress
        margin = self.curr_margin_start * (1.0 - progress)
        alpha = self.curr_alpha_start * (1.0 - progress) + self.curr_alpha_final * progress
        noise_std = self.curr_noise_std_start * (1.0 - progress)
        lambda_align = self.curr_lambda_align_start * (1.0 - progress)
        
        # Exponential decay for Beta (Primary Grid Disincentive)
        # beta = beta_final + (beta_start - beta_final) * exp(-3 * progress)
        beta = self.curr_beta_final + (self.curr_beta_start - self.curr_beta_final) * np.exp(-3 * progress)
        
        return {
            'progress': progress,
            'epsilon': epsilon,
            'margin': margin,
            'alpha': alpha,
            'beta': beta,
            'noise_std': noise_std,
            'lambda_align': lambda_align
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # SB3 handles Batch actions.
        raw_action = action.reshape(self.n_agents, 3)
        
        # --- PHASE 12: CURRICULUM UPDATES ---
        cur_params = self._get_curriculum_params()
        progress = cur_params['progress']
        
        if cur_params['noise_std'] > 1e-6:
            noise = self.rng.normal(0, cur_params['noise_std'], size=(self.n_agents, 3))
            raw_action[:, 1:] += noise[:, 1:]
            raw_action = np.clip(raw_action, -1.0, 1.0)
            
        # --- PHASE 10: ACTION MASKING (Consumers) ---
        for i in range(self.n_agents):
            if not self.nodes[i].is_prosumer:
                raw_action[i, 0] = 0.0 
                raw_action[i, 1] = np.clip(raw_action[i, 1], -1.0, 0.0)
        
        # Deadlock Breaking
        if self.market_history['steps_without_trade'] >= 12:
            noise = self.rng.normal(0, 0.05, size=raw_action.shape)
            raw_action = np.clip(raw_action + noise, -1.0, 1.0)
            
        state_data = self._step_physics_prep(raw_action)
        
        # 2. Market (Pass dynamic curriculum parameters and progress)
        market_results = self._step_market(state_data['safe_action'], cur_params['epsilon'], cur_params['margin'], progress)
        
        # 3. Grid & Rewards (Pass full curriculum context)
        obs, reward, info = self._step_grid_and_reward(state_data, market_results, cur_params)
        
        self.timestep_count += 1
        self.current_idx += 1
        
        # Check if we've reached the end of the dataset or max episode length
        done = False
        truncated = False
        if self.df is not None and self.current_idx >= len(self.df):
            truncated = True  # Episode truncated due to dataset end
        elif self.timestep_count >= self.max_steps:
            truncated = True  # Episode truncated due to max steps
        
        return obs, reward, done, truncated, info

    def _step_physics_prep(self, raw_action):
         # Get Data
        demands, pvs, co2 = self._get_current_data()
        
        # Autonomous Guard (Filter)
        current_socs = np.array([n.soc for n in self.nodes])
        state_for_guard = np.stack([demands, current_socs, pvs], axis=1) # (N, 3)
        
        safe_action_flat, guard_info = self.guard.process_intent(
            step=self.timestep_count,
            observations=self._get_obs(0,0), # Approx
            rl_actions=raw_action.flatten(),
            state=state_for_guard
        )
        safe_action = safe_action_flat.reshape(self.n_agents, 3)
        
        # Execute local physics (Battery)
        batt_action = safe_action[:, 0]
        node_results = []
        for i, node in enumerate(self.nodes):
            res = node.step(
                battery_action_kw=batt_action[i],
                current_demand_kw=demands[i],
                current_pv_kw=pvs[i],
                dt_hours=self.timestep_hours
            )
            node_results.append(res)
            
        return {
            'demands': demands,
            'pvs': pvs,
            'co2': co2[0] if isinstance(co2, np.ndarray) else co2, # Use temp or scalar
            'safe_action': safe_action, # Filtered action
            'node_results': node_results,
            'guard_info': guard_info
        }

    def _step_market(self, safe_action, epsilon=0.12, margin=0.0, progress=1.0):
        grid_req = safe_action[:, 1]
        price_bid = safe_action[:, 2]
        
        # Market Liberalization: No hard clipping.
        # Agents can bid negatives (pay to dump) or high prices (scarcity).
        # We only clip to physical sanity if needed (e.g. +/- 10.0), but let's leave unbounded.
        # price_bid = np.clip(price_bid, FEED_IN, RETAIL) <- REMOVED
        # Phase 5: Double Auction Logic
        # Actions: [Charge(0), Trade(1), Bid(2)]
        # Trade > 0: Seller (Ask), Trade < 0: Buyer (Bid)
        
        raw_actions = safe_action # (N, 3)
        trade_intents = raw_actions[:, 1]
        bid_prices = np.clip(raw_actions[:, 2], 0.0, 1.0) # Price perception
        
        # We need a reference "Market Price" or do we let them discover it?
        # The prompt says: "If an agent asks for a price higher than the grid price, the trade fails."
        # Actually, "Bid_Price action... If agent asks for price higher than grid... trades fail."
        # Let's interpret:
        # 1. Sellers submit Ask Price. Buyers submit Bid Price.
        # 2. Clearing: We sort Bids (desc) and Asks (asc). Match where Bid >= Ask.
        # 3. BUT, user prompt says: "If an agent asks for a price higher than the grid price, the trade fails."
        # implying implicit competition with Grid.
        # Let's simple P2P clearing:
        # - Buyers buy from P2P if Bid >= P2P_Ask and P2P_Ask < Grid_Retail
        # - Sellers sell to P2P if Ask <= P2P_Bid and P2P_Bid > Feed_In
        
        # Current MatchingEngine might need update, or we do pre-filtering here.
        # Let's standard "Double Auction" via MatchingEngine if it supports it, 
        # or we implement a simplified "Pairwise Check" here.
        
        # Simplified for RL Stability (as per prompt hint "simple Mid-Market -> Competitive"):
        # We will use the 'bid_prices' as their Limit Price.
        # Buyers: "I will pay at most $X".
        # Sellers: "I want at least $Y".
        
        # The Matching Engine determines a Clearing Price (e.g. Avg).
        # We just need to pass these limits to MatchingEngine.
        # Does internal MatchingEngine support limits? 
        # Let's assume standard 'matching_engine.match()' does Quantity matching.
        # We will wrap it with Price Check.
        
        # --- STEP 4: COMPLETE PRICE SMOOTHING ---
        total_market_demand = np.sum(np.abs(trade_intents[trade_intents < 0]))
        total_market_supply = np.sum(np.abs(trade_intents[trade_intents > 0]))
        
        base_p = 0.10
        k1 = 0.05
        eps = 1e-4
        price_raw = base_p + k1 * (total_market_demand / (total_market_supply + eps))
        
        # Temporal Smoothing (0.7 / 0.3)
        dynamic_price = 0.7 * self.prev_market_price + 0.3 * price_raw
        
        # Adaptive Correction
        if self.rolling_grid_dependency > 0.9:
            dynamic_price *= 0.9
            
        # Clamp and Persist
        dynamic_price = np.clip(dynamic_price, 0.05, 0.30)
        self.prev_market_price = dynamic_price
        
        retail_p, feed_in_p = self._get_grid_prices()
        
        # Selling Constraint: Don't undercut 80% of grid Feed-In Tariff (Viability Floor)
        # This prevents "toxic trades" where P2P price < 0.8 * Grid Sale price
        SELLER_FLOOR = 0.8 * feed_in_p
        
        # STEP 2: STABILIZE CONSUMER BID FORMULATION
        real_limit_prices = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            node = self.nodes[i]
            # Use raw action for 'bid perception' if prosumer, or special formula for consumer
            action_val = bid_prices[i] # Bounded [0, 1]
            
            if node.is_prosumer > 0.5: # Prosumer
                if trade_intents[i] > 0: # Seller
                    real_limit_prices[i] = max(dynamic_price, SELLER_FLOOR)
                else: # Buyer
                    real_limit_prices[i] = dynamic_price
            else: # Consumer
                # Rule-based bounded bid (Step 2)
                # sensitivity [0.8, 1.2] applied to a bounded action range
                bid = dynamic_price * (0.8 + 0.4 * action_val) * node.price_sensitivity
                real_limit_prices[i] = np.clip(bid, 0.05, 0.30)
        
        # Pass trades to Matching Engine
        bids_input = np.stack([trade_intents, real_limit_prices], axis=1)
        
        trades, market_price, net_grid_flow, match_info = self.matching_engine.match(
            bids_input, 
            grid_buy_price=retail_p, 
            grid_sell_price=feed_in_p,
            epsilon=epsilon,
            margin=margin,
            progress=progress,
            use_curriculum=self.use_curriculum
        )
        
        # Step 7: Calculate Liquidity Metric
        # liquidity = matched_volume / total_requested_volume
        liquidity = match_info['total_volume'] / max(1e-6, total_market_demand)
        
        # New matching engine handles everythinginternally
        failed_count = 0 
        
        # Update market history
        if market_price > 0:
            self.market_history['last_market_price'] = market_price
            self.market_history['prices'].append(market_price)
            
        success = float(match_info['total_volume'] > 1e-6)
        self.market_history['success_flags'].append(success)
        self.market_history['last_success_rate'] = np.mean(self.market_history['success_flags']) if self.market_history['success_flags'] else 0.0
        
        if success > 0:
            self.market_history['steps_without_trade'] = 0
        else:
            self.market_history['steps_without_trade'] += 1
            
        return {
            'trades': trades,
            'market_price': market_price,
            'bids_processed': bids_input,
            'failed_trades_count': failed_count, 
            'bid_prices': real_limit_prices,
            'trade_intents': trade_intents,
            'match_info': match_info,
            'price_std': float(np.std(real_limit_prices)),
            'unmatched_demand': float(np.sum(np.abs(trade_intents[trade_intents < 0])) - match_info['total_volume']),
            'liquidity': liquidity,
            'success': float(success),
            'curriculum_epsilon': epsilon,
            'curriculum_margin': margin
        }

    def _step_grid_and_reward(self, physics_state, market_results, cur_params):
        trades = market_results['trades'] # Net exchange (P2P + Grid)
        market_price = market_results['market_price']
        co2 = physics_state['co2'] # Carbon intensity
        node_results = physics_state['node_results']
        safe_action = physics_state['safe_action']
        retail_p, feed_in_p = self._get_grid_prices()
        
        # --- PHASE 12: DYNAMIC INCENTIVES ---
        beta = cur_params['beta']
        alpha = cur_params['alpha']
        lambda_align = cur_params['lambda_align']
        avg_market_price = self.market_history['last_market_price']
        
        # Calculate Coordination Alignment Reward
        # alignment = exp(-abs(bid - avg_market_price))
        # Use bids_processed which has the real limits
        bids = market_results['bid_prices']
        alignment_penalty = np.exp(-np.abs(bids - avg_market_price))
        alignment_reward = alignment_penalty * lambda_align

        # 1. Economic Coefficients (Unified with Phase 12 Curriculum)
        # beta and alpha are used directly below.
        SMOOTHING_COEFF = 0.02 
        CO2_PENAL_COEFF = 0.15 
        
        # 2. Grid & P2P Analysis
        # Ground Truth: Each agent's net physical requirement
        # node_results is a list of dicts from MicrogridNode.step()
        phys_net_load = np.array([res['net_load_kw'] for res in node_results])
        
        # P2P matched volume comes from the matching engine (cleared between agents)
        # We need to extract the P2P portion of the trades.
        # market_results['match_info']['total_volume'] is the aggregate P2P volume.
        total_p2p_volume = market_results.get('match_info', {}).get('total_volume', 0.0)
        
        # Determine each agent's p2p_trade (part of their net_load satisfied by P2P)
        # In a pro-rata uniform auction, we can derive this from market_results['trades']
        # which already contains [P2P + Bidded_Grid].
        # But it's safer to just allocate total_p2p_volume based on the matching engine's logic.
        
        p2p_agent_trades = np.zeros(self.n_agents)
        # trades_market = market_results['trades'] # This only includes bidded volume
        
        # For simplicity and correctness: 
        # Actual Grid Trade = Physical_Net_Load - (successful P2P trades)
        # We'll use the matching engine's P2P volume to ensure consistency
        
        # Identify who actually traded P2P (those who bid and were matched)
        # The matching engine 'trades' result includes both P2P and Grid-as-Backstop.
        # total_export_bidded = sum of positive market trades
        # total_import_bidded = sum of absolute negative market trades
        total_exp_b = np.sum(np.clip(trades, 0, None))
        total_imp_b = np.sum(np.abs(np.clip(trades, None, 0)))
        
        for i in range(self.n_agents):
            if trades[i] > 0: # Seller in market
                p2p_agent_trades[i] = trades[i] * (total_p2p_volume / total_exp_b if total_exp_b > 0 else 0)
            else: # Buyer in market
                p2p_agent_trades[i] = trades[i] * (total_p2p_volume / total_imp_b if total_imp_b > 0 else 0)
        
        # The ACTUAL energy exchange with the grid for agent i is:
        # grid_trade[i] = physical_net_load[i] - p2p_agent_trades[i]
        # BUT wait: net_load is (Dem - PV + Batt). P2P trade is positive for sell, negative for buy.
        # If net_load is -1.0 (surplus) and p2p_trade is 0.2 (sell), grid_export = 0.8.
        # If net_load is 1.0 (deficit) and p2p_trade is -0.3 (buy), grid_import = 0.7.
        # Formula: actual_grid_flow[i] = -phys_net_load[i] + p2p_agent_trades[i] 
        # (Using Matching Engine sign convention: + is export, - is import)
        
        actual_grid_flow = -phys_net_load + p2p_agent_trades
        
        total_export = np.sum(np.clip(actual_grid_flow, 0, None))
        total_import = np.sum(np.abs(np.clip(actual_grid_flow, None, 0)))
        
        # Update metrics for info
        self.total_grid_import += total_import
        self.total_grid_export += total_export
        self.total_p2p_volume += total_p2p_volume

        # 3. Component Calculations
        # 3a. Grid Import Penalty
        grid_import_kw = np.maximum(0.0, -actual_grid_flow)
        grid_import_penalty = grid_import_kw * beta
        
        # 3b. Action Smoothing Penalty
        # Applied to all 3 action components for stability
        action_diffs = np.abs(safe_action - self.prev_actions)
        smoothing_penalty = np.sum(action_diffs, axis=1) * SMOOTHING_COEFF
        
        # 3c. CO2 Penalty (based on grid import)
        # kg_co2 = grid_import_kwh * co2_intensity
        grid_import_kwh = grid_import_kw * self.timestep_hours
        co2_penalty = (grid_import_kwh * co2) * CO2_PENAL_COEFF
        
        # 3d. P2P Trading Bonus (+alpha per kWh)
        p2p_bonus = np.abs(p2p_agent_trades) * alpha
        
        # 3e. Overload Penalties
        line_overload_kw = max(0.0, max(total_export, total_import) - self.max_line_capacity_kw)
        overload_penalty_agent = np.ones(self.n_agents) * line_overload_kw * self.overload_multiplier / self.n_agents
        
        # 3f. Operational Penalties (SoC and Battery Wear)
        socs_final = np.array([r['soc'] for r in node_results])
        throughputs = np.array([r['throughput_delta'] for r in node_results])
        soc_penalty = (socs_final - 50.0)**2 * 0.001
        battery_wear_cost = throughputs * 0.02  # PHASE 14: reduced to allow arbitrage cycles
        
        # 3g. PHASE 14: 3-Layer ToU Battery Arbitrage Strategy
        hour = (self.current_idx % 24)
        is_solar_harvest = (6 <= hour < 16)
        is_prepeak       = (14 <= hour < 17)
        is_peak          = (17 <= hour < 21)
        batt_reward = np.zeros(self.n_agents)
        
        # 1-step solar lookahead
        next_gen = 0.0
        if self.df is not None:
            next_row = self.df.iloc[(self.current_idx + 1) % len(self.df)]
            next_gen = next_row.get('agent_0_pv_kw', 0.0)
        
        retail_p_local, _ = self._get_grid_prices()
        trade_intents = market_results.get('trade_intents', np.zeros(self.n_agents))
        
        for i in range(self.n_agents):
            soc_i   = socs_final[i]        # [0, 100] %
            pv_i    = physics_state['pvs'][i]
            dem_i   = physics_state['demands'][i]
            act_batt = safe_action[i, 0]   # > 0 = charge, < 0 = discharge
            net_node = pv_i - dem_i
            
            # ── Layer 1: Solar Harvest (6h–16h) ─────────────────────────
            # Charge from surplus PV while battery has headroom
            if is_solar_harvest and soc_i < 80.0 and (net_node > 0 or next_gen > 0.5) and act_batt > 0.05:
                batt_reward[i] += 0.06
            
            # ── Layer 2: Pre-Peak Top-Up (14h–17h) ──────────────────────
            # Ensure battery is well-charged before the expensive peak window
            if is_prepeak and soc_i < 60.0 and act_batt > 0.05:
                batt_reward[i] += 0.04
            
            # ── Layer 3: Peak Dispatch (17h–21h) ────────────────────────
            if is_peak:
                if soc_i > 20.0 and act_batt < -0.05:
                    # Reward discharging to avoid retail grid cost
                    batt_reward[i] += 0.10
                if soc_i > 20.0 and trade_intents[i] < -1e-6 and market_results.get('market_price', 0) > retail_p_local:
                    # Penalise importing from expensive grid when battery has reserves
                    batt_reward[i] -= 0.08
            
            # ── Buyer Rationality (unchanged) ───────────────────────────
            if trade_intents[i] < -1e-6 and market_results.get('market_price', 0.) > retail_p_local:
                batt_reward[i] -= 0.05 
        
        # 4. Final Aggregation
        raw_market_profits = trades * market_price
        
        # --- NEW: FUTURE-AWARE REWARD (Multi-Step Step 6) ---
        GAMMA_FUTURE = 0.05
        # prev_actions storage reused for profit? No, let's just use current.
        # But user suggestedgamma * expected_future_profit. 
        # For simplicity in 1-step MDP, we use a small reward persistence.
        
        # --- PHASE 13: SCIENTIFIC VALIDATION ---
        # Track quantities for scientific metrics
        self.reward_tracker.step_grid_import_quantities = grid_import_kw * self.timestep_hours
        self.reward_tracker.step_battery_quantities = throughputs * self.timestep_hours
        
        reward = self.reward_tracker.calculate_total_reward(
            profits=raw_market_profits,
            grid_import_penalties=grid_import_penalty,
            soc_penalties=soc_penalty,
            grid_overload_costs=overload_penalty_agent,
            battery_costs=battery_wear_cost + battery_wear_cost - batt_reward,
            smoothing_penalties=smoothing_penalty,
            co2_penalties=co2_penalty,
            p2p_bonuses=p2p_bonus,
            total_export_kw=total_export,
            traded_energy=np.abs(p2p_agent_trades),
            trade_intent=safe_action[:, 1],
            alpha=alpha,
            beta=beta,
            lambda_align=lambda_align,
            avg_market_price=avg_market_price,
            bids=np.stack([safe_action[:, 1], safe_action[:, 2]], axis=1), # [qty, price]
            use_alignment_reward=self.use_alignment_reward
        )
        
        # Calculate Phase 13 Metrics
        sci_metrics = self.reward_tracker.calculate_scientific_metrics(
            battery_degradation_rate=0.01, 
            co2_intensity=co2 if isinstance(co2, (float, int)) else 0.4
        )
        
        self.total_clean_profit += np.sum(sci_metrics["clean_profits"])
        self.total_economic_profit += np.sum(sci_metrics["economic_profits"])
        self.total_carbon_emissions += np.sum(sci_metrics["carbon_emissions"])
        
        # --- UPDATE ROLLING DEPENDENCY (Step 1) ---
        # Add a floor to demands to prevent infinite dependency (%) at scale
        total_net_demand = np.sum(physics_state['demands'])
        if total_net_demand > 0.1:
            # Cap at 1.0 (100%) to ensure it remains a valid 'dependency' metric even during battery charging
            current_dep = min(1.0, float(total_import / total_net_demand))
        else:
            current_dep = 0.0 
            
        self.rolling_grid_dep_buffer.append(current_dep)
        self.rolling_grid_dependency = np.mean(self.rolling_grid_dep_buffer)
        
        # --- LAGRANGIAN SAFETY PENALTY ---
        # Get current SoC values for all agents
        current_socs = np.array([node.soc for node in self.nodes])
        battery_caps = np.array([node.battery_capacity_kwh for node in self.nodes])

        # Total absolute network flow
        total_abs_flow_kw = max(total_export, total_import)
        
        lagrangian_violations = self.lagrangian.compute_violations(
            soc_values=current_socs,
            battery_capacities=battery_caps,
            line_flow_kw=total_abs_flow_kw,
            max_line_capacity_kw=self.max_line_capacity_kw,
            grid_voltage_kv=getattr(self, 'grid_voltage_kv', 0.4),
            line_resistance_ohm=getattr(self, 'line_resistance_ohms', 0.01),
        )
        lagrangian_penalty = self.lagrangian.compute_penalty(lagrangian_violations)
        self.lagrangian.record_step(lagrangian_violations)

        # Subtract Lagrangian penalty from total reward (Reduced weight)
        reward = reward - 0.5 * lagrangian_penalty

        # Add to info dict for TensorBoard logging (will be updated below)
        # --- END LAGRANGIAN ---
        
        # 5. Logging and Info Update
        profit_breakdown = self.reward_tracker.get_profit_breakdown()
        
        # --- SCIENTIFIC METRIC BIFURCATION (Fixed for Demo) ---
        market_price = market_results.get('market_price', 0.15)
        # Market Profit = Savings per kWh traded locally vs Grid
        retail_p = 0.20
        market_profit_usd = float(total_p2p_volume * (retail_p - market_price))
        
        # 2. Economic Profit: Community Savings - Battery Depreciation
        total_battery_wear = np.sum(battery_wear_cost)
        economic_profit_usd = float(market_profit_usd - total_battery_wear)
        
        # 3. Consumption tracking
        total_demand_kw = float(np.sum(physics_state['demands']))

        self.prev_actions = safe_action.copy()
        r_info = self.reward_tracker.get_info()
        
        info = {
            "market_price": market_price,
            "total_export": total_export,
            "total_import": total_import,
            "grid_capacity_limit": self.max_line_capacity_kw,
            "line_overload_kw": line_overload_kw,
            "p2p_volume_kwh_step": total_p2p_volume * self.timestep_hours,
            "total_demand_kw": float(total_demand_kw),
            "grid_import_kwh": float(np.sum(grid_import_kwh)),
            "carbon_intensity": co2,
            "clean_profit_usd": np.sum(sci_metrics["clean_profits"]),
            "economic_profit_usd": np.sum(sci_metrics["economic_profits"]),
            "carbon_emissions_kg": np.sum(sci_metrics["carbon_emissions"]),
            "cum_clean_profit": self.total_clean_profit,
            "cum_economic_profit": self.total_economic_profit,
            "cum_carbon_emissions": self.total_carbon_emissions,
            "trade_success_rate": self.market_history['last_success_rate'],
            "grid_dependency": float(self.rolling_grid_dependency),
            "p2p_volume": float(total_p2p_volume),
            "liquidity": market_results.get('liquidity', 0.0),
            "curriculum_epsilon": market_results.get('curriculum_epsilon', 0.15),
            "curriculum_margin": market_results.get('curriculum_margin', 0.0),
            "curriculum_beta": float(beta),
            "success": market_results.get('success', 0.0),
            "profit_breakdown": profit_breakdown,
            **r_info,
            **physics_state["guard_info"],
            "market_profit_usd": market_profit_usd,
            "failed_trades": market_results.get('failed_trades_count', 0),
            
            # Lagrangian Stats
            **self.lagrangian.get_lambda_info(),
            'lagrangian/violation_soc':     float(lagrangian_violations[0]),
            'lagrangian/violation_line':    float(lagrangian_violations[1]),
            'lagrangian/violation_voltage': float(lagrangian_violations[2]),
            'lagrangian/penalty_this_step': lagrangian_penalty
        }
        
        obs = self._get_obs(total_export, total_import)
        return obs, reward, info

    def _get_grid_prices(self):
        """
        Returns (Retail Price, Feed-In Tariff)
        Could be dynamic based on time of day.
        For now, fixed or simple ToU.
        """
        # Simple ToU: Peak 17-21
        hour = (self.current_idx % 24)
        if 17 <= hour < 21:
            return 0.50, 0.10 # High retail during peak
        return 0.20, 0.10


    def _get_obs(self, total_export, total_import):
        """
        Generates Observation Vector (Scalable Heterogeneous).
        Fixed size independent of N_agents.
        """
        dem, gen, weather = self._get_current_data()
        socs = np.array([n.soc for n in self.nodes])
        caps = np.array([n.battery_capacity_kwh for n in self.nodes])
        
        retail, feed_in = self._get_grid_prices()
        
        hour = (self.current_idx % 24)
        sin_time = np.sin(2 * np.pi * hour / 24.0)
        cos_time = np.cos(2 * np.pi * hour / 24.0)
        
        MAX_PRICE = 0.50
        retail_norm = retail / MAX_PRICE
        feed_in_norm = feed_in / MAX_PRICE
        
        # Global Market Stats (Step 4)
        total_market_dem = np.sum(dem)
        total_market_gen = np.sum(gen)
        avg_p = np.mean(self.market_history['prices']) if self.market_history['prices'] else 0.15
        
        # Batch observations for all agents
        all_obs = []
        for i in range(self.n_agents):
            node = self.nodes[i]
            
            # Local Features (Step 3)
            # [SoC, Is_Prosumer, Own_Demand, Own_Gen]
            soc_norm = socs[i] / caps[i] if caps[i] > 0 else 0.0
            local = [
                soc_norm, 
                node.is_prosumer, 
                dem[i], 
                gen[i]
            ]
            
            # Context & Time
            context = [retail_norm, feed_in_norm, sin_time, cos_time]
            
            # Global Market Stats (Step 4) - Scalable averages
            global_metrics = [
                avg_p / MAX_PRICE,
                total_market_dem / 50.0, # Norm to line capacity approx
                total_market_gen / 50.0,
                self.market_history['last_success_rate']
            ]
            
            # Forecast (if enabled)
            forecast_block = []
            if self.forecast_horizon > 0:
                # Agent-specific demand forecast
                for h in range(1, self.forecast_horizon + 1):
                    row = self.df.iloc[(self.current_idx + h) % len(self.df)]
                    f_dem = row.get(f"agent_{i%4}_demand_kw", 0.0)
                    f_gen = row.get(f"agent_{i%4}_pv_kw", 0.0) if node.is_prosumer else 0.0
                    forecast_block.extend([f_dem, f_gen])
            
            agent_obs = np.concatenate([local, context, global_metrics, forecast_block])
            all_obs.append(agent_obs)
            
        return np.array(all_obs)

    def _get_obs_legacy(self, total_export, total_import):
        """Phase 3 Observer (Physical Units)."""
        dem, pv, co2 = self._get_current_data()
        socs = np.array([n.soc for n in self.nodes])
        retail, feed_in = self._get_grid_prices()
        
        obs_list = []
        for i in range(self.n_agents):
            base = [dem[i], socs[i], pv[i], total_export, total_import, co2, retail, feed_in]
            
            if self.forecast_horizon > 0:
                forecasts = []
                for h in range(self.forecast_horizon):
                     n_dem = self.rng.normal(0, 0.05 * dem[i] + 0.05)
                     n_pv = self.rng.normal(0, 0.05 * pv[i] + 0.05)
                     f_dem = max(0, dem[i] + n_dem)
                     f_pv = max(0, pv[i] + n_pv)
                     f_price = retail 
                     forecasts.extend([f_dem, 0.1, f_pv, f_price]) # Legacy 4-var forecast
                base.extend(forecasts)
            obs_list.extend(base)
            
        return np.array(obs_list, dtype=np.float32)

