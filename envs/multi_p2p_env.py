# envs/multi_p2p_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any
from market.matching_engine import MatchingEngine
from utils.metrics import gini_coefficient

class MultiP2PEnergyEnv(gym.Env):
    """
    Multi-prosumer P2P environment.
    - The action is a vector of length N: power (kW) for each prosumer.
      Positive -> sell to market, Negative -> buy from market (or charge battery).
    - Observation = concatenated per-agent features: [gen, demand, soc, last_price] for each agent.
    - Uses a simple pro-rata matching engine and grid balancing.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=5, timestep_hours=1.0, episode_len=24, seed=None, config: Dict=None):
        super().__init__()
        self.n = n_agents
        self.timestep = timestep_hours
        self.episode_len = episode_len
        self.rng = np.random.RandomState(seed)
        # Per-agent parameters (can be vectorized or scalar)
        cfg = config or {}
        self.battery_capacity = cfg.get('battery_capacity_kwh', 10.0)  # per agent
        self.max_power = cfg.get('max_power_kw', 5.0)
        self.charge_eff = cfg.get('charge_eff', 0.95)
        self.grid_price_baseline = cfg.get('grid_price_baseline', 0.2)  # $/kWh baseline
        self.grid_carbon = cfg.get('grid_carbon', 0.6)  # kgCO2/kWh for grid
        self.renewable_carbon = cfg.get('renewable_carbon', 0.05)

        # Observations: for each agent [gen, demand, soc, last_price]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0] * self.n, dtype=np.float32)
        obs_high = np.array([cfg.get('max_gen_kw', 10.0),
                             cfg.get('max_demand_kw', 10.0),
                             self.battery_capacity,
                             100.0] * self.n, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: continuous power per agent in [-max_power, +max_power]
        self.action_space = spaces.Box(low=np.array([-self.max_power]*self.n, dtype=np.float32),
                                       high=np.array([self.max_power]*self.n, dtype=np.float32),
                                       dtype=np.float32)

        # State
        self.t = 0
        self.soc = np.full(self.n, self.battery_capacity * 0.5, dtype=np.float32)
        self.gen_profile = None
        self.demand_profile = None
        self.last_price = np.full(self.n, self.grid_price_baseline, dtype=np.float32)

        # Matching engine
        self.matching_engine = MatchingEngine()

        # Logging / metrics
        self.episode = 0
        self.total_profits = np.zeros(self.n, dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.episode += 1
        self.soc = np.full(self.n, self.battery_capacity * 0.5, dtype=np.float32)
        self.last_price = np.full(self.n, self.grid_price_baseline, dtype=np.float32)
        # Sample simple daily profiles (sinusoidal for generation, demand)
        hours = np.arange(self.episode_len)
        peak_gen = 5.0
        peak_demand = 6.0
        ph = self.rng.uniform(0, 2*np.pi, size=self.n)
        self.gen_profile = (peak_gen * np.maximum(0, np.sin((2*np.pi/24)*(hours + ph[:,None])))).clip(0, peak_gen)
        self.demand_profile = (peak_demand * (0.5 + 0.5*np.cos((2*np.pi/24)*(hours + ph[:,None])))).clip(0, peak_demand)
        # Make profiles shape (n, episode_len)
        if self.gen_profile.ndim == 2:
            # already (n, len) if ph was broadcast
            pass
        else:
            self.gen_profile = np.tile(self.gen_profile, (self.n, 1))
            self.demand_profile = np.tile(self.demand_profile, (self.n, 1))
        self.total_profits = np.zeros(self.n, dtype=np.float32)
        info = {}
        return self._get_obs(), info

    def step(self, action):
        """
        action: np.array shape (n,) continuous power in kW
        """
        action = np.array(action, dtype=np.float32).reshape(self.n)
        action = np.clip(action, -self.max_power, self.max_power)

        gen = self._get_gen_t()
        demand = self._get_demand_t()

        # Action interpretation:
        # If action > 0: agent wants to sell 'action' kW to market (or discharge battery to satisfy own demand first)
        # If action < 0: agent wants to buy '-action' kW from market (or charge battery)
        # We simulate battery charging/discharging given action and local surplus/deficit.
        # Compute available local surplus (gen - demand)
        local_surplus = gen - demand  # could be negative
        # Agents may use battery to meet deficit or store surplus; action decides net to/from market.
        # For simplicity: battery changes are result of residual after local consumption and market trades.
        # Prepare orders for matching engine: positive sells, negative buys
        orders = action.copy()

        # Matching engine: returns per-agent traded amounts and clearing price, grid imbalance
        trades, clearing_price, grid_import = self.matching_engine.match(orders)
        # trades: positive = sold by agent to P2P, negative = bought by agent from P2P (kW)
        # grid_import: positive means grid supplied (kW), negative means grid absorbed surplus.

        # Update battery SOC due to local balancing and leftover after trading
        # For each agent, net_local = gen - demand - traded (traded >0 means sold away)
        net_local_after_trade = gen - demand - trades
        # If net_local_after_trade > 0 -> surplus -> charge battery (or curtail)
        # If net_local_after_trade < 0 -> deficit -> discharge battery (or import from grid)
        soc = self.soc.copy()
        for i in range(self.n):
            net = net_local_after_trade[i] * self.timestep  # kWh for timestep
            if net > 0:
                # charge battery
                charge = min(net * self.charge_eff, self.battery_capacity - soc[i])
                soc[i] += charge
                leftover = net - charge / self.charge_eff
                # leftover curtailed (ignored)
            else:
                # need energy -> discharge battery first
                need = -net
                discharge = min(need / self.charge_eff, soc[i])
                soc[i] -= discharge
                remaining_need = need - discharge * self.charge_eff
                # remaining_need must be satisfied by grid import (if any)
                # handled by grid_import accounting (matching engine gives grid_import sum)
        # Update SOC
        self.soc = soc

        # Compute rewards per agent:
        rewards = np.zeros(self.n, dtype=np.float32)
        emissions = np.zeros(self.n, dtype=np.float32)
        profits = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            # Energy from trades (kW) * timestep => kWh
            traded_kwh = trades[i] * self.timestep
            revenue = max(0.0, traded_kwh) * clearing_price  # sold => revenue
            cost = max(0.0, -traded_kwh) * clearing_price  # bought => cost
            # Grid interactions cost/emissions for remaining net after trades and battery
            # approximate grid usage: if net_local_after_trade < 0 after battery then grid supplies
            grid_kwh = 0.0
            net = net_local_after_trade[i] * self.timestep
            # compute net after battery discharge/charge approximated by soc changes
            # simpler: if demand exceeded gen+discharge -> grid_kwh = remaining_need
            # This is already approximated earlier; for reward we assume grid_kwh = max(0, -net)
            if net < 0:
                # some portion may have been covered by battery; approximate taken from SOC delta
                # compute change in soc this step:
                # rarely exact but sufficient for prototyping
                # if SOC decreased compared to previous step -> that discharge covered some
                # For simplicity assume grid_kwh = max(0, -net - (previous_soc - current_soc))
                pass
            # Approximate emissions: traded energy from P2P uses renewable fraction first
            # We'll assume trades use peer energy (mixed); use small carbon factor for renewables
            traded_emission = abs(traded_kwh) * self.renewable_carbon
            # Grid emissions: if agent net after trades is negative (needs energy), approximate remaining need as grid
            grid_kwh_agent = max(0.0, - (gen[i] * self.timestep - demand[i] * self.timestep - traded_kwh))
            grid_emission = grid_kwh_agent * self.grid_carbon
            emissions[i] = traded_emission + grid_emission
            profits[i] = revenue - cost
            # battery degradation cost (penalize cycles): proportional to abs(change in soc)
            # approximate: small cost per kWh cycled
            # can't get previous soc easily here; ignore or add small constant
            batt_cost = 0.01 * abs(traded_kwh)  # small placeholder
            rewards[i] = profits[i] - 0.1 * emissions[i] - batt_cost

        # Fairness penalty: compute Gini across cumulative profits (including this step)
        cum_profits = self.total_profits + profits
        gini = gini_coefficient(cum_profits)
        # apply fairness penalty to each agent's reward (same scaled penalty)
        fairness_lambda = 0.2  # hyperparam: scale of fairness penalty
        fairness_penalty_each = fairness_lambda * gini
        rewards = rewards - fairness_penalty_each

        # update totals
        self.total_profits += profits
        self.last_price = np.full(self.n, clearing_price, dtype=np.float32)

        self.t += 1
        terminated = (self.t >= self.episode_len)
        truncated = False  # No truncation in this environment
        info: Dict[str, Any] = {
            'clearing_price': float(clearing_price),
            'trades_kw': trades,
            'grid_import_kw': float(grid_import),
            'profits': profits,
            'emissions': emissions,
            'gini': float(gini),
            'rewards_per_agent': rewards  # Store per-agent rewards in info for analysis
        }
        obs = self._get_obs()
        # Aggregate rewards to scalar for stable_baselines3 (centralized policy)
        reward_scalar = float(np.sum(rewards))
        return obs, reward_scalar, terminated, truncated, info

    def _get_obs(self):
        # build observation vector concatenating per-agent features
        gen = self._get_gen_t()
        demand = self._get_demand_t()
        obs = []
        for i in range(self.n):
            obs.extend([float(gen[i]), float(demand[i]), float(self.soc[i]), float(self.last_price[i])])
        return np.array(obs, dtype=np.float32)

    def _get_gen_t(self):
        # get generation for time t for each agent (n,)
        if self.t < self.episode_len:
            return self.gen_profile[:, self.t]
        else:
            # if beyond length, return zeros
            return np.zeros(self.n, dtype=np.float32)

    def _get_demand_t(self):
        if self.t < self.episode_len:
            return self.demand_profile[:, self.t]
        else:
            return np.zeros(self.n, dtype=np.float32)

    def render(self, mode='human'):
        print(f"t={self.t}, soc={self.soc}, last_price={self.last_price}")

    def close(self):
        pass
