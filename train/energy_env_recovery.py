
"""
Phase 5 Recovery: Liquidity-First P2P Energy Trading Environment

Key Changes from Original:
1. REMOVED price bidding from action space (now 2D: battery_action, p2p_quantity)
2. Automatic Mid-Market Rate (MMR) pricing: (Grid_Buy + Grid_Sell) / 2
3. Forced P2P matching: Automatic pairing of buyers and sellers
4. Peer demand observations: Agents see total surplus/deficit of other agents
5. Positive reward shaping: P2P participation bonus, healthy SoC bonus
6. Reduced penalty coefficients for stability

This refactor aims to fix policy collapse and jumpstart P2P trading activity.
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional

# Local modules
try:
    from train.autonomous_guard import AutonomousGuard
    from train.reward_tracker_recovery import RewardTrackerRecovery
    from simulation.microgrid import MicrogridNode
except ImportError:
    from .autonomous_guard import AutonomousGuard
    from .reward_tracker_recovery import RewardTrackerRecovery
    from ..simulation.microgrid import MicrogridNode


class EnergyMarketEnvRecovery(gym.Env):
    """
    Phase 5 Recovery Environment with Liquidity-First P2P Trading.
    
    Simplified market mechanism to enable RL agents to learn P2P trading.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self,
                 n_agents: int = 4,
                 data_file: str = "processed_hybrid_data.csv",
                 random_start_day: bool = True,
                 enable_ramp_rates: bool = True,
                 enable_losses: bool = True,
                 forecast_horizon: int = 4,
                 forecast_noise_std: float = 0.05,
                 diversity_mode: bool = True,
                 seed: int = 42):
        super(EnergyMarketEnvRecovery, self).__init__()
        
        self.n_agents = n_agents
        self.data_file = data_file
        self.random_start_day = random_start_day
        self.enable_ramp_rates = enable_ramp_rates
        self.enable_losses = enable_losses
        self.forecast_horizon = forecast_horizon
        self.forecast_noise_std = forecast_noise_std
        self.diversity_mode = diversity_mode
        self.timestep_hours = 1.0  # Hourly simulation

        # State tracking (Moved to top to fix max_steps dependency)
        self.current_idx = 0
        self.day_start_idx = 0
        self.timestep_count = 0
        self.max_steps = 24 * 7  # 1 week episodes
        
        # Load Data (Now safe to call)
        self._load_data()
        
        # Setup Agents (Physics Nodes)
        self._setup_agents()
        
        # Guard (using max specs for safety)
        self.guard = AutonomousGuard(
            n_agents=self.n_agents,
            battery_capacity_kwh=62.0,  # Max (EV)
            battery_max_charge_kw=7.0,   # Max (EV)
            timestep_hours=self.timestep_hours
        )
        
        # === ACTION SPACE: SIMPLIFIED ===
        # REMOVED: price_bid (was dimension 2)
        # NOW: [battery_action_kw, p2p_quantity_request_kw]
        # Range: [-1, 1] normalized, scaled inside step()
        # Flattened: (N * 2,)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(n_agents * 2,),  # Changed from 3 to 2
            dtype=np.float32
        )
        
        # === OBSERVATION SPACE: ENHANCED ===
        # Base features (7): SoC, Retail, FeedIn, SinTime, CosTime, TotalExp, TotalImp
        # Weather (2): Temperature, WindSpeed
        # Agent Type One-Hot (4): Solar, Wind, EV, Standard
        # Forecast (2 * H): [Demand, Generation] for each horizon step
        # NEW: Peer Demand (1): Total surplus/deficit of other agents
        n_base = 7
        n_weather = 2
        n_type = 4
        n_forecast = 2 * self.forecast_horizon
        n_peer = 1  # NEW: Peer demand feature
        
        n_obs_features = n_base + n_weather + n_type + n_forecast + n_peer
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_agents * n_obs_features,),
            dtype=np.float32
        )
        
        # History buffer for smoothing penalty
        self.prev_actions = np.zeros((n_agents, 2))  # Changed from 3 to 2
        
        # RNG
        self.rng = np.random.default_rng(seed)
        
        # P2P Trading Stats (for monitoring)
        self.p2p_trades_count = 0
        self.p2p_volume_total = 0.0
    
    def _setup_agents(self):
        """Setup heterogeneous agent archetypes"""
        self.nodes = []
        
        # Params mapping based on MicrogridNode(node_id, kap, max_kw, eff)
        
        # Agent 0: Solar (High Gen, Low Battery)
        n0 = MicrogridNode(
            node_id=0,
            battery_capacity_kwh=5.0 if self.diversity_mode else 10.0,
            battery_max_charge_kw=2.5 if self.diversity_mode else 5.0,
            battery_eff=0.9
        )
        if self.enable_ramp_rates: n0.max_ramp_kw = 2.5 # Example ramp limit
        self.nodes.append(n0)
        
        # Agent 1: Wind (High Gen, Low Battery)
        n1 = MicrogridNode(
            node_id=1,
            battery_capacity_kwh=5.0,
            battery_max_charge_kw=2.5,
            battery_eff=0.9
        )
        if self.enable_ramp_rates: n1.max_ramp_kw = 2.5
        self.nodes.append(n1)
        
        # Agent 2: EV (Mobile, Huge Battery, High Power)
        n2 = MicrogridNode(
            node_id=2,
            battery_capacity_kwh=62.0, # Tesla Model 3 Long Range
            battery_max_charge_kw=7.0, # Home Charger
            battery_eff=0.9
        )
        if self.enable_ramp_rates: n2.max_ramp_kw = 7.0
        self.nodes.append(n2)
        
        # Agent 3: Standard (Baseline)
        n3 = MicrogridNode(
            node_id=3,
            battery_capacity_kwh=10.0,
            battery_max_charge_kw=5.0,
            battery_eff=0.9
        )
        if self.enable_ramp_rates: n3.max_ramp_kw = 5.0
        self.nodes.append(n3)

    def _load_data(self):
        # Assuming data matches expected format
        if not os.path.exists(self.data_file):
            # Fallback for testing path mainly
            if os.path.exists(os.path.join("..", self.data_file)):
                self.data_file = os.path.join("..", self.data_file)
            elif os.path.exists(os.path.join("data", self.data_file)):
                 self.data_file = os.path.join("data", self.data_file)
        
        try:
            self.df = pd.read_csv(self.data_file)
            # Ensure required columns exist, fill if missing (robust loading)
            required = [
                'agent_0_demand', 'agent_0_pv',
                'agent_1_demand', 'agent_1_wind',
                'agent_2_demand', 'agent_2_pv',
                'agent_3_demand', 'agent_3_pv',
                'temperature_2m', 'windspeed_100m'
            ]
            for col in required:
                if col not in self.df.columns:
                    self.df[col] = 0.0
                    
            self.max_idx = len(self.df) - self.max_steps - 1
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing if file missing
            dates = pd.date_range(start='1/1/2017', periods=2000, freq='H')
            self.df = pd.DataFrame(index=dates)
            for col in required:
                 self.df[col] = np.random.rand(2000)
            self.max_idx = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        # Select start day
        if self.random_start_day:
            # Ensure we don't go out of bounds
            low = 0
            high = max(1, self.max_idx - 1)
            self.day_start_idx = self.rng.integers(low, high)
        else:
            self.day_start_idx = 0
            
        self.current_idx = self.day_start_idx
        self.timestep_count = 0
        self.p2p_trades_count = 0
        self.p2p_volume_total = 0.0
        self.prev_actions = np.zeros((self.n_agents, 2))
        
        # Reset Agents
        for node in self.nodes:
            node.reset()
            # Randomize SoC
            node.soc = self.rng.uniform(0, node.battery_capacity_kwh)
            
        return self._get_obs(0.0, 0.0), {}

    def _get_grid_prices(self):
        # Simple ToU
        # Hour of day can be derived from index if we assume start=Jan1 00:00
        # or simplified: 8am-8pm Peak, else Off-peak
        hour = (self.current_idx % 24)
        
        if 14 <= hour <= 20: # Peak
            sell = 0.08
            buy = 0.30 
        elif 8 <= hour < 14: # Shoulder
            sell = 0.06
            buy = 0.20
        else: # Off-peak
            sell = 0.05
            buy = 0.15
            
        return buy, sell

    def _get_current_data(self):
        """Fetch current timestep data from dataset"""
        row = self.df.iloc[self.current_idx]
        
        demands = []
        generations = []
        
        # Agent 0: Solar
        demands.append(row.get('agent_0_demand', 0.0))
        generations.append(row.get('agent_0_pv', 0.0))
        
        # Agent 1: Wind
        demands.append(row.get('agent_1_demand', 0.0))
        generations.append(row.get('agent_1_wind', 0.0))
        
        # Agent 2: EV
        demands.append(row.get('agent_2_demand', 0.0))
        generations.append(row.get('agent_2_pv', 0.0))
        
        # Agent 3: Standard
        demands.append(row.get('agent_3_demand', 0.0))
        generations.append(row.get('agent_3_pv', 0.0))
        
        # Weather
        temp = row.get('temperature_2m', 20.0)
        wind = row.get('windspeed_100m', 5.0)
        
        return np.array(demands), np.array(generations), np.array([temp, wind])

    def _apply_ev_constraints(self):
        """Update EV capacity based on schedule"""
        # EV (Agent 2) is Away 8am - 5pm
        hour = (self.current_idx % 24)
        ev_node = self.nodes[2]
        
        if 8 <= hour < 17:
             # Away: Reduced capacity (e.g. 2kWh buffer or disconnected)
             # If disconnected, cap=0? Or small buffer.
             ev_node.battery_capacity_kwh = 2.0
             # Clip SoC if needed (physically it's elsewhere, but locally limited)
             ev_node.soc = min(ev_node.soc, ev_node.battery_capacity_kwh)
        else:
             # Home: Full Capacity (62kWh)
             ev_node.battery_capacity_kwh = 62.0

    def _get_obs(self, total_export, total_import):
        # Calculate derived features
        hour = (self.current_idx % 24)
        sin_time = np.sin(2 * np.pi * hour / 24)
        cos_time = np.cos(2 * np.pi * hour / 24)
        
        buy, sell = self._get_grid_prices()
        
        # Get Data
        demands, generations, weather = self._get_current_data()
        
        # Calculate Peer Demand (Total Net Load of others)
        # Surplus = Gen - Demand. (Positive = Surplus)
        # We want to see if MARKET has surplus.
        current_net_loads = generations - demands 
        # Note: This ignores battery state, purely generation/demand surplus
        
        # Normalize features roughly
        weather_norm = weather / [40.0, 20.0] # Temp max 40, Wind max 20
        
        obs_list = []
        for i in range(self.n_agents):
            node = self.nodes[i]
            
            # Peer Demand for agent i = Sum of others' net load
            peer_surplus = np.sum(current_net_loads) - current_net_loads[i]
            peer_demand = -peer_surplus # Positive means peer deficit (demand)
            # Scale
            peer_demand /= 10.0 
            
            # Forecasts (Simple lookahead from df)
            forecasts = []
            for h in range(1, self.forecast_horizon + 1):
                idx = min(self.current_idx + h, self.max_idx)
                row = self.df.iloc[idx]
                if i == 0: d=row.get('agent_0_demand',0); g=row.get('agent_0_pv',0)
                elif i == 1: d=row.get('agent_1_demand',0); g=row.get('agent_1_wind',0)
                elif i == 2: d=row.get('agent_2_demand',0); g=row.get('agent_2_pv',0)
                elif i == 3: d=row.get('agent_3_demand',0); g=row.get('agent_3_pv',0)
                forecasts.extend([d/5.0, g/5.0]) # Scale approx
            
            # Agent Type One-Hot
            type_vec = np.zeros(4)
            type_vec[i] = 1.0
            
            base = [
                node.soc / 62.0, # Norm by max possible cap
                buy,
                sell,
                sin_time,
                cos_time,
                total_export / 20.0,
                total_import / 20.0
            ]
            
            # Concatenate all features
            full_obs = np.concatenate([
                base,
                weather_norm,
                type_vec,
                forecasts,
                [peer_demand]  # NEW
            ])
            
            obs_list.extend(full_obs)
        
        return np.array(obs_list, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep with simplified P2P market.
        
        Action space (per agent): [battery_action, p2p_quantity_request]
        - battery_action: [-1, 1] -> scaled to [-max_charge_kw, +max_charge_kw]
        - p2p_quantity_request: [-1, 1] -> scaled to [-max_trade_kw, +max_trade_kw]
          - Positive: Want to SELL (have surplus)
          - Negative: Want to BUY (have deficit)
        """
        # Reshape from flat to (N, 2)
        raw_action = action.reshape(self.n_agents, 2)
        
        # Apply EV constraints
        self._apply_ev_constraints()
        
        # Get current data
        demands, generations, weather = self._get_current_data()
        
        # === STEP 1: Battery Physics ===
        # Scale battery actions to physical units
        battery_actions_kw = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            max_charge = self.nodes[i].battery_max_charge_kw
            battery_actions_kw[i] = raw_action[i, 0] * max_charge
        
        # Execute battery step
        node_results = []
        for i, node in enumerate(self.nodes):
            res = node.step(
                battery_action_kw=battery_actions_kw[i],
                current_demand_kw=demands[i],
                current_pv_kw=generations[i],
                dt_hours=self.timestep_hours
            )
            node_results.append(res)
        
        # === STEP 2: Simplified P2P Market (Liquidity-First) ===
        # Scale P2P quantity requests
        p2p_requests_kw = raw_action[:, 1] * 10.0  # Max 10 kW trade request
        
        # Automatic Mid-Market Rate (MMR)
        retail, feed_in = self._get_grid_prices()
        mmr_price = (retail + feed_in) / 2.0
        
        # Forced Matching Algorithm
        trades_kw, p2p_volume = self._execute_p2p_matching(p2p_requests_kw, mmr_price)
        
        # === STEP 3: Grid Interaction ===
        # After P2P, remaining needs go to grid
        # trades_kw: Positive = Bought (Import), Negative = Sold (Export)
        # Net Load (Physical) - Trade (Import) = Remaining Need
        
        net_loads = np.array([r['net_load_kw'] for r in node_results])
        grid_flows = net_loads - trades_kw  # Remaining after P2P
        
        total_export = np.sum(np.clip(grid_flows, 0, None)) # Wait, Export is Negative grid_flow?
        # Convention check:
        # Net Load > 0: Deficit (Need Power).
        # Grid Flow > 0: Import.
        # Grid Flow < 0: Export.
        
        # Calculation:
        # total_export should be sum of absolute negative flows
        total_export = np.sum(np.abs(np.clip(grid_flows, None, 0)))
        total_import = np.sum(np.clip(grid_flows, 0, None))
        
        # Grid overload penalty
        # (Optional)
        
        # === REWARD CALCULATION ===
        rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            r = 0.0
            
            # 1. Financial Outcome (Simulated)
            # P2P Settlement
            p2p_qty = trades_kw[i]
            # Cost = P2P_Import * Price
            # Rev = P2P_Export * Price
            if p2p_qty > 0: # Bought
                r -= p2p_qty * mmr_price
            else: # Sold
                r += abs(p2p_qty) * mmr_price
                
            # Grid Settlement
            gf = grid_flows[i]
            if gf > 0: # Import
                r -= gf * retail
            else: # Export
                r += abs(gf) * feed_in
                
            # 2. Shaping: Encouraging P2P
            if abs(p2p_qty) > 0.1:
                r += 0.05 # Small bonus for participating
            
            # 3. Shaping: SoC Health
            # Determine target SoC (e.g., 50%)
            soc_norm = self.nodes[i].soc / self.nodes[i].battery_capacity_kwh
            dist = abs(soc_norm - 0.5)
            r -= 0.01 * dist # Stay near middle
            
            # 4. Peer Demand Matching Bonus
            # If I sold when peers needed, bonus?
            # Implicit in P2P price vs FeedIn.
            
            rewards[i] = r
            
            # History
            self.prev_actions[i] = raw_action[i]
            
        # Global stats
        self.p2p_volume_total += p2p_volume
        self.current_idx += 1
        self.timestep_count += 1
        
        done = False
        truncated = False
        if self.current_idx >= self.max_idx or self.timestep_count >= self.max_steps:
            truncated = True
            
        obs = self._get_obs(total_export, total_import)
        
        info = {
            'p2p_volume': p2p_volume,
            'p2p_trades': self.p2p_trades_count,
            'clearing_price': mmr_price
        }
        
        return obs, rewards, done, truncated, info

    def _execute_p2p_matching(self, requests_kw, price):
        """
        Simple Pro-Rata Matching.
        requests_kw: +ve (Sell), -ve (Buy)? 
        Check docstring: "Positive: Want to SELL". "Negative: Want to BUY".
        """
        # Separate Supply (Sell) and Demand (Buy)
        supply_indices = np.where(requests_kw > 0)[0]
        demand_indices = np.where(requests_kw < 0)[0]
        
        total_supply = np.sum(requests_kw[supply_indices])
        total_demand = np.abs(np.sum(requests_kw[demand_indices]))
        
        traded_volume = min(total_supply, total_demand)
        
        # Allocation Ratios
        sell_ratio = traded_volume / total_supply if total_supply > 1e-6 else 0.0
        buy_ratio = traded_volume / total_demand if total_demand > 1e-6 else 0.0
        
        # Execute
        final_trades = np.zeros_like(requests_kw)
        
        # Sellers sell pro-rata
        final_trades[supply_indices] = -1.0 * requests_kw[supply_indices] * sell_ratio 
        # Note: If I Sell, I export. Export implies grid_flow decrease (negative).
        # My convention above:
        # trades_kw: Positive = Bought (Import). Negative = Sold (Export).
        # requests_kw: Positive = Sell.
        # So "Sold" trade should be negative.
        # yes, -1.0 * positive * ratio = negative. Correct.
        
        # Buyers buy pro-rata
        final_trades[demand_indices] = np.abs(requests_kw[demand_indices]) * buy_ratio
        # requests_kw is negative. abs() is positive.
        # final_trades positive = Import. Correct.
        
        return final_trades, traded_volume
