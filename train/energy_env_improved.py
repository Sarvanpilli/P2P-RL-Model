# energy_env_improved.py
# Patched environment: stronger price sensitivity & overload penalty + 3-step naive forecast in obs

# Suppress Gym deprecation warnings - MUST be first, before any imports
import os
os.environ["GYM_NOTICES"] = "0"  # Disable gym_notices package warnings

import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym_notices")

import numpy as np
import gymnasium as gym
from gymnasium import spaces

def transmission_loss(power_kw, alpha=1e-4, beta=1e-6):
    p = np.asarray(power_kw, dtype=float)
    return alpha * np.abs(p) + beta * (p ** 2)

def battery_deg_cost(throughput_kwh, dod, k_throughput=0.02, k_dod=0.5):
    return k_throughput * throughput_kwh + k_dod * (dod ** 2)

class EnergyMarketEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 n_agents=4,
                 timestep_hours=1.0,
                 max_line_capacity_kw=200.0,
                 per_agent_max_kw=200.0,
                 alpha_loss=1e-4,
                 beta_loss=1e-6,
                 battery_capacity_kwh=50.0,
                 battery_max_charge_kw=25.0,
                 battery_roundtrip_eff=0.95,
                 battery_deg_k_throughput=0.02,
                 battery_deg_k_dod=0.5,
                 base_price=0.12,
                 price_slope=0.01,             # increased sensitivity
                 overload_multiplier=50.0,     # stronger overload penalty
                 forecast_horizon=3,           # 3-step naive forecast
                 seed=None):
        super().__init__()

        self.n_agents = int(n_agents)
        self.timestep_hours = float(timestep_hours)
        self.max_line_capacity_kw = float(max_line_capacity_kw)
        self.per_agent_max_kw = float(per_agent_max_kw)
        self.alpha_loss = float(alpha_loss)
        self.beta_loss = float(beta_loss)

        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.battery_max_charge_kw = float(battery_max_charge_kw)
        self.battery_eff = float(battery_roundtrip_eff)
        self.batt_k_throughput = float(battery_deg_k_throughput)
        self.batt_k_dod = float(battery_deg_k_dod)

        self.base_price = float(base_price)
        self.price_slope = float(price_slope)
        self.overload_multiplier = float(overload_multiplier)

        self.forecast_horizon = int(forecast_horizon)

        # state: per-agent [demand, soc, pv]
        self.state = np.zeros((self.n_agents, 3), dtype=np.float32)

        # action: per-agent [battery_kw, grid_trade_kw]
        action_low = np.array([-self.battery_max_charge_kw, -self.per_agent_max_kw], dtype=np.float32)
        action_high = np.array([ self.battery_max_charge_kw,  self.per_agent_max_kw], dtype=np.float32)
        self.action_space = spaces.Box(low=np.tile(action_low, (self.n_agents,)),
                                       high=np.tile(action_high, (self.n_agents,)),
                                       dtype=np.float32)

        # observation per-agent: [demand, soc, pv, total_export, total_import,
        #                         pv_f1..pv_fH, dem_f1..dem_fH]
        h = self.forecast_horizon
        per_len = 3 + 2 + h + h  # demand,soc,pv + total_export,total_import + pv forecasts + demand forecasts
        high_val = np.finfo(np.float32).max / 8.0
        obs_low = np.zeros(per_len, dtype=np.float32)
        obs_high = np.array([high_val, max(1.0, self.battery_capacity_kwh), high_val,
                             high_val, high_val] + [high_val]*h + [high_val]*h, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.tile(obs_low, (self.n_agents,)),
                                            high=np.tile(obs_high, (self.n_agents,)),
                                            dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        demands = self.rng.uniform(0.0, 50.0, size=(self.n_agents,))
        soc = self.rng.uniform(0.2 * self.battery_capacity_kwh,
                               0.8 * self.battery_capacity_kwh,
                               size=(self.n_agents,))
        pv = self.rng.uniform(0.0, 40.0, size=(self.n_agents,))
        self.state[:, 0] = demands
        self.state[:, 1] = soc
        self.state[:, 2] = pv
        self.timestep_count = 0
        return self._get_obs(), {}

    def _naive_forecast(self, array, h):
        # simple deterministic upward/downward linear forecast with small noise factor
        # array shape: (n_agents,)
        if h == 0:
            # Return empty array with shape (n_agents, 0)
            return np.empty((array.shape[0], 0), dtype=np.float32)
        forecasts = []
        for step in range(1, h+1):
            factor = 1.0 + 0.02 * step  # small trend multiplier (tunable)
            forecasts.append(array * factor)
        return np.stack(forecasts, axis=1)  # shape (n_agents, h)

    def _get_obs(self, total_export=0.0, total_import=0.0):
        # compute naive forecasts for pv and demand
        h = self.forecast_horizon
        pv_fore = self._naive_forecast(self.state[:,2], h)  # (n_agents, h)
        dem_fore = self._naive_forecast(self.state[:,0], h)
        per_agent = np.zeros((self.n_agents, 3 + 2 + h + h), dtype=np.float32)
        per_agent[:, 0:3] = self.state
        per_agent[:, 3] = float(total_export)
        per_agent[:, 4] = float(total_import)
        per_agent[:, 5:5+h] = pv_fore
        per_agent[:, 5+h:5+h+h] = dem_fore
        return per_agent.flatten().astype(np.float32)

    def _apply_action_limits(self, battery_kw, grid_trade_kw):
        battery_kw = np.clip(battery_kw, -self.battery_max_charge_kw, self.battery_max_charge_kw)
        grid_trade_kw = np.clip(grid_trade_kw, -self.per_agent_max_kw, self.per_agent_max_kw)
        return battery_kw, grid_trade_kw

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (2 * self.n_agents,)
        action = action.reshape(self.n_agents, 2)
        battery_action = action[:, 0].copy()
        grid_trade = action[:, 1].copy()
        battery_action, grid_trade = self._apply_action_limits(battery_action, grid_trade)

        demand = self.state[:, 0]
        soc = self.state[:, 1].copy()
        pv = self.state[:, 2]

        battery_charge_kw = np.clip(battery_action, 0, None)
        battery_discharge_kw = -np.clip(battery_action, None, 0)

        effective_batt_charge_kw = np.zeros(self.n_agents, dtype=float)
        effective_batt_discharge_kw = np.zeros(self.n_agents, dtype=float)
        batt_throughput_kwh = np.zeros(self.n_agents, dtype=float)
        batt_dod = np.zeros(self.n_agents, dtype=float)

        for i in range(self.n_agents):
            max_charge_possible_kwh = max(0.0, self.battery_capacity_kwh - soc[i])
            max_charge_possible_kw = max_charge_possible_kwh / max(self.timestep_hours, 1e-9)
            actual_charge_kw = min(battery_charge_kw[i], max_charge_possible_kw)

            max_discharge_possible_kwh = soc[i]
            max_discharge_possible_kw = max_discharge_possible_kwh / max(self.timestep_hours, 1e-9)
            actual_discharge_kw = min(battery_discharge_kw[i], max_discharge_possible_kw)

            if actual_charge_kw > 0:
                energy_charged_kwh = actual_charge_kw * self.timestep_hours * (self.battery_eff ** 0.5)
                soc[i] += energy_charged_kwh
                effective_batt_charge_kw[i] = actual_charge_kw
                batt_throughput_kwh[i] += abs(energy_charged_kwh)
            if actual_discharge_kw > 0:
                energy_discharged_kwh = actual_discharge_kw * self.timestep_hours * (self.battery_eff ** 0.5)
                soc[i] -= energy_discharged_kwh
                effective_batt_discharge_kw[i] = actual_discharge_kw
                batt_throughput_kwh[i] += abs(energy_discharged_kwh)

            soc[i] = np.clip(soc[i], 0.0, self.battery_capacity_kwh)
            dod_i = 1.0 - (soc[i] / self.battery_capacity_kwh)
            batt_dod[i] = np.clip(dod_i, 0.0, 1.0)

        supply_from_local_kw = pv + effective_batt_discharge_kw
        demand_from_local_kw = demand + effective_batt_charge_kw

        intended_injection_kw = grid_trade.copy()
        total_export_kw = np.sum(np.clip(intended_injection_kw, 0, None))
        total_import_kw = np.sum(np.abs(np.clip(intended_injection_kw, None, 0)))

        export_overload = max(0.0, total_export_kw - self.max_line_capacity_kw)
        import_overload = max(0.0, total_import_kw - self.max_line_capacity_kw)
        line_overload_kw = max(export_overload, import_overload)

        curtailment = np.zeros(self.n_agents, dtype=float)
        if export_overload > 0 and total_export_kw > 0:
            exporters = np.where(intended_injection_kw > 0)[0]
            export_vals = intended_injection_kw[exporters]
            scale = self.max_line_capacity_kw / (total_export_kw + 1e-12)
            new_vals = export_vals * scale
            curtail = export_vals - new_vals
            intended_injection_kw[exporters] = new_vals
            curtailment[exporters] = curtail
        if import_overload > 0 and total_import_kw > 0:
            importers = np.where(intended_injection_kw < 0)[0]
            import_vals = -intended_injection_kw[importers]
            scale = self.max_line_capacity_kw / (total_import_kw + 1e-12)
            new_vals = import_vals * scale
            curtail = import_vals - new_vals
            intended_injection_kw[importers] = -new_vals
            curtailment[importers] = curtail

        total_export_kw_after = np.sum(np.clip(intended_injection_kw, 0, None))
        total_import_kw_after = np.sum(np.abs(np.clip(intended_injection_kw, None, 0)))

        total_supply_kw = np.sum(supply_from_local_kw) + total_export_kw_after
        total_demand_kw = np.sum(demand_from_local_kw) + total_import_kw_after
        net_shortage_kw = max(0.0, total_demand_kw - total_supply_kw)
        market_price = self.base_price + self.price_slope * net_shortage_kw

        losses_kw = transmission_loss(intended_injection_kw, alpha=self.alpha_loss, beta=self.beta_loss)
        effective_grid_injection_kw = intended_injection_kw.copy()
        exporters_mask = intended_injection_kw > 0
        importers_mask = intended_injection_kw < 0
        effective_grid_injection_kw[exporters_mask] = intended_injection_kw[exporters_mask] - losses_kw[exporters_mask]
        effective_grid_injection_kw[importers_mask] = intended_injection_kw[importers_mask] - losses_kw[importers_mask]

        batt_deg_costs = np.array([battery_deg_cost(batt_throughput_kwh[i], batt_dod[i],
                                                    k_throughput=self.batt_k_throughput, k_dod=self.batt_k_dod)
                                   for i in range(self.n_agents)])

        energy_hours = self.timestep_hours
        rewards = np.zeros(self.n_agents, dtype=float)
        for i in range(self.n_agents):
            exchanged_kwh = effective_grid_injection_kw[i] * energy_hours
            rewards[i] += exchanged_kwh * market_price
            if curtailment[i] > 0:
                rewards[i] -= curtailment[i] * energy_hours * market_price
            if line_overload_kw > 0:
                denom = np.sum(np.abs(intended_injection_kw)) + 1e-12
                contribution = abs(intended_injection_kw[i]) / denom
                rewards[i] -= contribution * self.overload_multiplier * (1.0 + 0.01 * line_overload_kw)
            rewards[i] -= batt_deg_costs[i]

        # update state simple random walk
        self.state[:, 0] = np.clip(self.state[:, 0] + self.rng.normal(0, 1.0, size=self.n_agents), 0.0, 500.0)
        self.state[:, 1] = soc
        self.state[:, 2] = np.clip(self.state[:, 2] + self.rng.normal(0, 2.0, size=self.n_agents), 0.0, 200.0)
        self.timestep_count += 1

        info = {
            "market_price": market_price,
            "loss_kw": losses_kw,
            "line_overload_kw": line_overload_kw,
            "export_overload_kw": export_overload,
            "import_overload_kw": import_overload,
            "curtailment_kw": curtailment,
            "battery_deg_costs": batt_deg_costs,
            "battery_throughput_kwh": batt_throughput_kwh,
            "intended_injection_kw": intended_injection_kw,
            "effective_grid_injection_kw": effective_grid_injection_kw,
            "total_export_kw_before": total_export_kw,
            "total_import_kw_before": total_import_kw,
            "total_export_kw_after": total_export_kw_after,
            "total_import_kw_after": total_import_kw_after,
            "rewards_per_agent": rewards.copy(),  # Store per-agent rewards for analysis
        }

        obs = self._get_obs(total_export=total_export_kw, total_import=total_import_kw)
        terminated = False
        truncated = False
        # Aggregate rewards to scalar for Stable-Baselines3 (sum of all agent rewards)
        reward_scalar = float(np.sum(rewards))
        return obs, reward_scalar, terminated, truncated, info

    def render(self):
        print(f"t={self.timestep_count} per-agent states:")
        for i in range(self.n_agents):
            print(f" A{i}: demand={self.state[i,0]:.2f}kW soc={self.state[i,1]:.2f}kWh pv={self.state[i,2]:.2f}kW")
        print("----")
