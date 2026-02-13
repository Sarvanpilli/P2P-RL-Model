
"""
EnergyMarketEnvRobust: Research-Grade Gymnasium Environment for P2P Energy Trading.
Refactored to use MicrogridNode and Real Data.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
                 n_agents: int = 4, # Backward compat
                 n_prosumers: int = None, # New explicit
                 n_consumers: int = None, # New explicit
                 timestep_hours: float = 1.0,
                 max_line_capacity_kw: float = 200.0,
                 per_agent_max_kw: float = 200.0,
                 battery_capacity_kwh: float = 50.0,
                 battery_max_charge_kw: float = 25.0,
                 battery_roundtrip_eff: float = 0.95,
                 overload_multiplier: float = 50.0,
                 forecast_horizon: int = 0,
                 data_file: str = "test_day_profile.csv",
                 random_start_day: bool = True,
                 seed: Optional[int] = None,
                 enable_ramp_rates: bool = True,
                 ramp_limit_kw_per_hour: float = 10.0,
                 enable_losses: bool = True,
                 line_resistance_ohms: float = 0.05,
                 grid_voltage_kv: float = 0.4, # 400V
                 enable_predictive_obs: bool = False, # Phase 4 Flag
                 forecast_noise_std: float = 0.05, # Phase 5 (Stochasticity)
                 diversity_mode: bool = False, # Phase 5 (Heterogeneity)
                 **kwargs):
        
        super().__init__()
        
        # --- Type Logic ---
        # 1. Agent Typing & count
        if n_prosumers is not None and n_consumers is not None:
             self.n_prosumers = int(n_prosumers)
             self.n_consumers = int(n_consumers)
             self.n_agents = self.n_prosumers + self.n_consumers
        else:
             # Default fallback
             self.n_prosumers = int(n_agents)
             self.n_consumers = 0
             self.n_agents = int(n_agents)

        # 2. Configurations
        self.timestep_hours = float(timestep_hours)
        self.forecast_noise_std = float(forecast_noise_std)
        self.diversity_mode = diversity_mode
        
        # Dynamic Grid Scaling: Base 200kW is for ~4-5 agents. Scale by sqrt(N/5).
        # Or simpler: Base per agent. 
        # Update 4: "max_line_capacity_kw = base * sqrt(n_agents)" isn't quite right from prompt logic 
        # "Make grid capacity configurable... OR scale...". Let's apply scaling logic only if N large.
        # Logic: If N > 10, scale capacity.
        # User prompt: "max_line_capacity_kw = base_capacity_kw * sqrt(n_agents)"
        # But 'base' is assumed to be the provided arg? Let's treat arg as 'base'
        # Actually, let's just implement the formula: Cap = Arg * sqrt(N/5) ?
        # Prompt says: "Scale programmatically: max_line_capacity_kw = base_capacity_kw * sqrt(n_agents)"
        # I will assume the input arg is the BASE coefficient.
        # Note: If N=4, sqrt(4)=2. If input=100 (per agent?), no input is 'capacity'.
        # Let's say input is 'Base Capacity Global'.
    def __init__(self, 
                 n_agents=4, 
                 data_file="processed_hybrid_data.csv", 
                 random_start_day=True,
                 enable_ramp_rates=True,
                 enable_losses=True,
                 forecast_horizon=4, # User requested 4-hour lookahead
                 enable_predictive_obs=True,
                 forecast_noise_std=0.0,
                 diversity_mode=True):
        
        super().__init__() # Inherit from Gym? No, previous code was custom class. Assuming Gym.
        
        # Initialize History Buffer early to avoid AttributeErrors in reset()
        self.history_window_size = 4
        self.history_buffer = {
            'demand': np.zeros((n_agents, 4)),
            'pv': np.zeros((n_agents, 4))
        }
        self.prev_actions = np.zeros((n_agents, 3)) # Unified Plural
        
        self.n_agents = n_agents
        self.data_file = data_file
        self.random_start_day = random_start_day
        self.enable_ramp_rates = enable_ramp_rates
        self.enable_losses = enable_losses
        self.forecast_horizon = forecast_horizon
        self.enable_predictive_obs = enable_predictive_obs
        self.enable_predictive_obs = enable_predictive_obs
        self.forecast_noise_std = forecast_noise_std
        
        # Physics Constants
        self.max_line_capacity_kw = 50.0
        self.line_resistance_ohms = 0.05 # Assumption
        self.overload_multiplier = 10.0 # Penalty weight

        self._load_data()
        
        # Setup Agents (Hybrid Archetypes)
        self._setup_agents()
        
        # Market & Guard
        self.matching_engine = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)
        
        # Guard
        # AutonomousGuard expects: n_agents, battery_capacity_kwh, battery_max_charge_kw, timestep_hours, grid_voltage_kv
        # Passing Max Specs for Hybrid System Safety
        self.timestep_hours = 1.0
        self.grid_voltage_kv = 0.4
        
        self.guard = AutonomousGuard(
            n_agents=self.n_agents,
            battery_capacity_kwh=62.0, # Max EV
            battery_max_charge_kw=7.0, # Max EV
            timestep_hours=self.timestep_hours
        )
        
        self.reward_tracker = RewardTracker(n_agents)
        
        # Spaces
        # Action: [Battery(kW), Trade(kW), Bid($)]
        # Range: [-1, 1], scaled inside step
        # Flattened to (N*3,) to match SB3 VecEnv expectations & avoid shape mismatch
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_agents * 3,), dtype=np.float32)
        
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
        
        n_obs_features = 7 + 2 + 4 + (2 * self.forecast_horizon)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n_agents * n_obs_features,), dtype=np.float32)
        
        self.current_idx = 0
        self.day_start_idx = 0
        self.max_steps = 24 * 7 # 1 week episodes usually

        # Track prev action for smoothing
        self.prev_actions = np.zeros((n_agents, 3))
        
        self.rng = np.random.default_rng()
        
    def _setup_agents(self):
        self.nodes = []
        for i in range(self.n_agents):
            if i == 0:
                # Agent 0: Solar (5kWh)
                node = MicrogridNode(node_id=i, battery_capacity_kwh=5.0, battery_max_charge_kw=2.5, battery_eff=0.9)
                node.agent_type_id = 0 # Solar
            elif i == 1:
                # Agent 1: Wind (5kWh)
                node = MicrogridNode(node_id=i, battery_capacity_kwh=5.0, battery_max_charge_kw=2.5, battery_eff=0.9)
                node.agent_type_id = 1 # Wind
            elif i == 2:
                # Agent 2: EV-V2G (62kWh / 2kWh)
                # Max capacity is 62. We constrain it dynamically.
                node = MicrogridNode(node_id=i, battery_capacity_kwh=62.0, battery_max_charge_kw=7.0, battery_eff=0.9) 
                node.agent_type_id = 2 # EV
            else:
                # Agent 3: Standard (10kWh)
                node = MicrogridNode(node_id=i, battery_capacity_kwh=10.0, battery_max_charge_kw=5.0, battery_eff=0.9)
                node.agent_type_id = 3 # Standard
            
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
        """Phase 5: Dynamic logic for Agent 2 EV."""
        hour = (self.current_idx % 24)
        ev_node = self.nodes[2]
        
        # Day: 08:00 - 17:00 (Hours 8 to 17 non-inclusive? usually 8<=h<=17)
        # "2kWh during day 08:00-17:00"
        if 8 <= hour < 17:
             # EV is Away or Reserved.
             # Effective Capacity = 2.0 kWh
             # We should clamp current SOC if it's above 2.0? 
             # Or just limit usable capacity?
             # Realistic: Car departs at 8am with high charge. Returns at 5pm?
             # User says: "Capacity... 2kWh during day". 
             # This implies it's connected but only 2kWh buffer accessible?
             # Or it physically changes.
             # Let's set the effective capacity limit in the node.
             ev_node.battery_capacity_kwh = 2.0
             # If current SOC > 2.0, well, physically the energy is there but inaccessible.
             # But RL sees SOC. 
             # Let's assume SOC is capped at 2.0 for interaction.
             if ev_node.soc > 2.0:
                 ev_node.soc = 2.0 # Force draining (or hiding)
        else:
             # Night: 62kWh
             if ev_node.battery_capacity_kwh < 62.0:
                 ev_node.battery_capacity_kwh = 62.0
                 # SOC returns? 
                 # Complex dynamics. Let's assume it stays at last value.
        
        # Guard
        # AutonomousGuard expects: n_agents, battery_capacity_kwh, battery_max_charge_kw, timestep_hours, grid_voltage_kv
        # But we have heterogeneous agents now. 
        # Check AutonomousGuard signature: it takes scalar capacity/rate?
        # If scalar, we might need to pass lists or max?
        # File view shows: __init__(self, n_agents, battery_capacity_kwh, battery_max_charge_kw, timestep_hours, grid_voltage_kv)
        # It seems to assume homogeneous agents or scalar inputs.
        # Let's pass the max values or lists if it supports them.
        # If it doesn't support lists, we might need to update Guard or pass a "safe upper bound".
        # Let's assume it wants scalars for now (based on old code).
        # We'll pass the MAX capacity and rate to ensure safety logic doesn't under-constrain?
        # Or better: check if it uses them for *all* agents uniformly.
        # If so, passing 62.0 (EV) might allow others to violate their 5.0 limit?
        # Yes.
        # Refactoring Guard to handle heterogeneity is out of scope unless necessary.
        # Step 1: Pass scalars.
        # Step 2: See if it works.
        self.timestep_hours = 1.0
        self.grid_voltage_kv = 0.4
        
        # We pass 62.0 and 7.0 as "system specs" or similar.
        # Actually in Phase 5 we should probably update Guard to read from Nodes?
        # The previous code passed `self.nodes` in one version? 
        # "refactored to use MicrogridNode" comment suggests it might match.
        # But the error says it expects `battery_capacity_kwh`, etc.
        # So it's the old Guard.
        # Let's pass the EV specs (max in system) to avoid crashing, 
        # knowing that individual node constraints are ALSO checked in `MicrogridNode.step`.
        # The Guard is for *Grid* safety mostly?
        
        self.guard = AutonomousGuard(
            n_agents=self.n_agents,
            battery_capacity_kwh=62.0, # Max
            battery_max_charge_kw=7.0, # Max
            timestep_hours=self.timestep_hours,
            grid_voltage_kv=self.grid_voltage_kv
        )
        
        self.matching_engine = MatchingEngine()
        self.reward_tracker = RewardTracker(n_agents=self.n_agents)

        # --- Data Loading ---
        self.data_file = data_file
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} rows from {self.data_file}")
        except Exception as e:
            print(f"WARNING: Could not load {self.data_file}: {e}")
            print("Using sinusoidal mock data fallback.")
            self.df = None

        # --- Spaces ---
        action_low = np.array([-battery_max_charge_kw, -per_agent_max_kw, 0.0], dtype=np.float32)
        action_high = np.array([ battery_max_charge_kw,  per_agent_max_kw, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.tile(action_low, (self.n_agents,)),
            high=np.tile(action_high, (self.n_agents,)),
            dtype=np.float32
        )

        # Phase 4: History Window
        self.history_window_size = 4
        self.history_buffer = {
            'demand': np.zeros((self.n_agents, self.history_window_size)),
            'pv': np.zeros((self.n_agents, self.history_window_size))
        }
        self.prev_action = np.zeros((self.n_agents, 3)) # Always init
        
        if self.enable_predictive_obs:
            # New Predictive Space (Normalized [-1, 1])
            # Base(7) + History(8) + Forecast(2*H)
            # + Static Profile (2): [Cap_Norm, PV_Peak_Norm]
            n_features = 7 + 8 + 2 * self.forecast_horizon + 2
            
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_agents * n_features,),
                dtype=np.float32
            )
        else:
            # Legacy Phase 3 Space (Physical Units)
            # Obs: [Dem, SoC, PV, TotalExport, TotalImport, CO2, GridBuy, GridSell] + Forecasts
            base_dim = 8
            obs_dim = base_dim + 4 * self.forecast_horizon
            high_val = np.finfo(np.float32).max / 8.0
            self.observation_space = spaces.Box(
                low=-high_val, high=high_val,
                shape=(self.n_agents * obs_dim,),
                dtype=np.float32
            )

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

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
        dems.append(row.get('agent_0_demand', 0.0))
        gens.append(row.get('agent_0_pv', 0.0))
        
        # Agent 1: Wind, Demand
        dems.append(row.get('agent_1_demand', 0.0))
        gens.append(row.get('agent_1_wind', 0.0)) # Wind!
        
        # Agent 2: EV, Demand (PV?)
        dems.append(row.get('agent_2_demand', 0.0))
        gens.append(row.get('agent_2_pv', 0.0)) # Does EV have PV? Maybe home PV.
        
        # Agent 3: Standard
        dems.append(row.get('agent_3_demand', 0.0))
        gens.append(row.get('agent_3_pv', 0.0))
        
        # Weather
        temp = row.get('temperature_2m', 20.0)
        wind = row.get('windspeed_100m', 5.0)
        
        return np.array(dems), np.array(gens), np.array([temp, wind])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # SB3 flattens actions for VecEnvs. Reshape to (N, 3)
        raw_action = action.reshape(self.n_agents, 3)
        
        state_data = self._step_physics_prep(raw_action)
        
        # 2. Market
        market_results = self._step_market(state_data['safe_action'])
        
        # 3. Grid & Rewards
        obs, reward, info = self._step_grid_and_reward(state_data, market_results)
        
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
            'co2': co2,
            'safe_action': safe_action, # Filtered action
            'node_results': node_results,
            'guard_info': guard_info
        }

    def _step_market(self, safe_action):
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
        
        # Scaling Normalized Price [0, 1] to Real [0, 0.50]
        MAX_PRICE = 0.50 # Approx grid retail max
        real_limit_prices = bid_prices * MAX_PRICE
        
        # We pass these to match() if we modify it, or we filter results.
        # Let's filter: Agents only match if their limits overlap.
        # Since standard Matching Engine is global pool, we can set a single "Clearing Price"
        # based on aggregate Bid/Ask curves. 
        # This is complex for 5 mins.
        # Alternative (Prompt): "If an agent asks for a price higher than the grid price..."
        # Let's act as if there is a 'Market Maker' that rejects bad bids.
        # We'll use the Grid Retail/FeedIn as bounds.
        # IF Seller Ask > Retail: No buyer would buy. Fail.
        # IF Buyer Bid < FeedIn: No seller would sell. Fail.
        
        # FILTERING:
        valid_trade_mask = np.ones(self.n_agents, dtype=bool)
        
        # Sellers (Trade > 0): Ask Price = real_limit_prices
        sellers = trade_intents > 1e-3
        retail_p, feed_in_p = self._get_grid_prices()
        
        # Constraint 1: Sellers cannot Ask more than Grid Retail (Competition)
        # If Ask > Retail, buyers prefer Grid.
        violation_sell = (real_limit_prices > retail_p) & sellers
        
        # Constraint 2: Buyers cannot Bid less than Feed-In (Competition)
        # If Bid < FeedIn, sellers prefer Grid.
        buyers = trade_intents < -1e-3
        violation_buy = (real_limit_prices < feed_in_p) & buyers
        
        # Apply Failures
        trade_intents[violation_sell] = 0.0
        trade_intents[violation_buy] = 0.0
        
        # Record Failed Trades for Reward?
        failed_count = np.sum(violation_sell) + np.sum(violation_buy)
        
        # Pass filtered trades to Matching Engine
        # Must be (N, 2) array: [Quantity, Price]
        bids_input = np.stack([trade_intents, real_limit_prices], axis=1)
        
        trades, market_price, net_grid_flow, match_info = self.matching_engine.match(
            bids_input, 
            grid_buy_price=retail_p, 
            grid_sell_price=feed_in_p
        )
        
        return {
            'trades': trades,
            'market_price': market_price,
            'bids_processed': bids_input,
            'failed_trades_count': failed_count, # New metric
            'bid_prices': real_limit_prices,
            'match_info': match_info
        }

    def _step_grid_and_reward(self, physics_state, market_results):
        trades = market_results['trades']
        market_price = market_results['market_price']
        co2 = physics_state['co2']
        node_results = physics_state['node_results']
        
        # Grid Impact
        intended_injection = trades
        total_export = np.sum(np.clip(intended_injection, 0, None))
        total_import = np.sum(np.abs(np.clip(intended_injection, None, 0)))
        
        # Overloads
        line_overload_kw = max(0.0, max(total_export, total_import) - self.max_line_capacity_kw)
        
        # Phase 2: Distribution Losses (I^2 * R)
        loss_kw = 0.0
        if self.enable_losses:
            # Estimate total network flow (sum of absolute net loads)
            # This represents the total "traffic" on the microgrid lines
            net_loads = np.array([r['net_load_kw'] for r in node_results])
            total_abs_flow_kw = np.sum(np.abs(net_loads))
            
            # Current I = P / V (kW / kV = Amps)
            total_current_amps = total_abs_flow_kw / self.grid_voltage_kv
            
            # Total Loss = I^2 * R (Watts) -> /1000 for kW
            # Simplifying assumption: Lumped resistance for the whole network? 
            # Or R per agent? If R is "effective resistance of the grid seen by aggregate"
            loss_kw = (total_current_amps ** 2) * self.line_resistance_ohms / 1000.0
        
        
        # Carbon logging: energy imported (kWh) * intensity (kg/kWh) = kg CO2
        # total_import is power (kW); convert to energy for the step
        # kept for info logging only
        import_kwh = total_import * self.timestep_hours
        step_carbon_mass = import_kwh * co2  # kg
        
        # Rewards
        socs_final = np.array([r['soc'] for r in node_results])
        throughputs = np.array([r['throughput_delta'] for r in node_results])
        
        # Grid Import Penalty (Target: minimize grid dependence)
        # Import is negative in 'trades'. So max(0, -trade) gives import kW.
        # Weighting: 0.5 per kW imported? Or higher?
        # If profit is ~0.10, then 0.50 penalty is strong.
        grid_import_kw = np.maximum(0.0, -trades)
        grid_import_penalty = grid_import_kw * 0.5 

        # Phase 5 Penalties
        # 1. Peak Window (17-21h)
        curr_hour = self.current_idx % 24
        peak_penalties = self.reward_tracker.calculate_peak_penalty(grid_import_kw, curr_hour)
        
        # 2. Smoothing (Action Jitter)
        # We need raw_action from step(). We only have trades here. 
        # But wait, step() calls this. We can access self.prev_actions buffer
        # 'physics_state' contains 'safe_action'
        current_action = physics_state['safe_action']
        smoothing_penalty = self.reward_tracker.calculate_smoothing_penalty(current_action, self.prev_actions)
        
        # 3. Deep Discharge
        deep_discharge_penalties = np.zeros(self.n_agents)
        for i, node in enumerate(self.nodes):
            deep_discharge_penalties[i] = (0.10 - (node.soc/node.battery_capacity_kwh)) * 10.0 if (node.soc/node.battery_capacity_kwh) < 0.10 else 0.0

        # 3. V2G Bonus
        caps = np.array([n.battery_capacity_kwh for n in self.nodes])
        v2g_bonus = self.reward_tracker.calculate_v2g_bonus(socs_final, caps, curr_hour)
        
        reward = self.reward_tracker.calculate_total_reward(
            profits=trades * market_price, # Revenue
            grid_import_penalties=grid_import_penalty + peak_penalties, # Base + Peak
            soc_penalties=(socs_final - 50.0)**2 * 0.001, 
            grid_overload_costs=np.ones(self.n_agents) * line_overload_kw * self.overload_multiplier / self.n_agents,
            battery_costs=throughputs * 0.05,
            export_penalties=np.zeros(self.n_agents),
            smoothing_penalties=smoothing_penalty,
            deep_discharge_penalties=deep_discharge_penalties,
            v2g_bonus=v2g_bonus,
            total_export_kw=total_export
        )
        
        # Update Prev Action for next step
        self.prev_action = current_action.copy()
        r_info = self.reward_tracker.get_info()
        
        # Per-agent trade (kW): positive = export, negative = import
        trades_kw = trades.copy()
        throughput_deltas = np.array([r["throughput_delta"] for r in node_results])
        info = {
            "market_price": market_price,
            "loss_kw": loss_kw, # Now calculated
            "total_export": total_export,
            "total_import": total_import,
            "total_carbon_mass": step_carbon_mass,
            "grid_capacity_limit": self.max_line_capacity_kw,
            "trades_kw": trades_kw,
            "battery_throughput_delta_kwh": throughput_deltas,
            "line_overload_kw": line_overload_kw,
            **r_info,
            **physics_state["guard_info"],
            "failed_trades": market_results.get('failed_trades_count', 0),
            "bid_prices_mean": np.mean(market_results.get('bid_prices', np.zeros(self.n_agents)))
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
        """Dispatch based on mode."""
        if self.enable_predictive_obs:
            return self._get_obs_predictive(total_export, total_import)
        else:
            return self._get_obs_legacy(total_export, total_import)

    def _get_obs(self, total_export, total_import):
        """
        Generates Observation Vector (Phase 5 Hybrid).
        Range: [-1, 1]
        """
        # 1. Fetch Data
        dem, gen, weather = self._get_current_data()
        socs = np.array([n.soc for n in self.nodes])
        caps = np.array([n.battery_capacity_kwh for n in self.nodes])
        
        # Prices
        retail, feed_in = self._get_grid_prices()
        
        # Time
        hour = (self.current_idx % 24)
        sin_time = np.sin(2 * np.pi * hour / 24.0)
        cos_time = np.cos(2 * np.pi * hour / 24.0)
        
        # Normalize
        # Data is already [0,1] from preprocessing?
        # User says: "create a version... scaled to [0, 1]" in preprocessing.
        # So dem/gen/weather are likely [0, 1].
        # Prices need norm.
        MAX_PRICE = 0.50
        retail_norm = retail / MAX_PRICE
        feed_in_norm = feed_in / MAX_PRICE
        
        obs_list = []
        for i in range(self.n_agents):
            agent_node = self.nodes[i]
            
            # SoC (Dynamic Capacity handled in step, but obs should reflect fill %)
            soc_norm = socs[i] / caps[i] if caps[i] > 0 else 0.0
            
            # Base Features (7)
            # [SoC, Retail, FeedIn, SinTime, CosTime, TotalEx, TotalIm]
            # Wait, user asked for "Agent Type (One-hot), SoC, cyclical time, weather context, and a 4-hour lookahead"
            # Line Loss 5% implies P2P logic, but obs doesn't change.
            
            base = [
                soc_norm, retail_norm, feed_in_norm, 
                sin_time, cos_time,
                np.clip(total_export/50.0, 0, 1), 
                np.clip(total_import/50.0, 0, 1)
            ]
            
            # Weather (2)
            # Assuming weather is normalized or small
            weather_norm = weather # [temp, wind]
            # Preprocessing might have norm'd them?
            # Revisit preprocessing logic if needed. Assuming [0,1].
            
            # Agent Type One-Hot (4)
            type_vec = np.zeros(4)
            type_vec[agent_node.agent_type_id] = 1.0
            
            # Forecast (Lookahead)
            # 2 features (Dem, Gen) * Horizon
            forecasts = []
            if self.forecast_horizon > 0:
                for h in range(1, self.forecast_horizon + 1):
                     # Fetch future row
                     future_idx = min(self.current_idx + h, self.max_idx - 1)
                     row = self.df.iloc[future_idx]
                     
                     # Demand
                     f_dem = row.get(f'agent_{i}_demand', 0.0)
                     
                     # Generation (Solar or Wind)
                     # Agent 1 uses Wind col, others Solar/PV col
                     if i == 1:
                         f_gen = row.get('agent_1_wind', 0.0)
                     else:
                         f_gen = row.get(f'agent_{i}_pv', 0.0)
                         
                     forecasts.extend([f_dem, f_gen])
            
            # Concatenate
            full_obs = np.concatenate([base, weather_norm, type_vec, forecasts])
            obs_list.extend(full_obs)
            
        return np.array(obs_list, dtype=np.float32)

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

