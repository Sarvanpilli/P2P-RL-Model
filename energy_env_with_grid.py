# energy_env_with_grid_gymnasium.py
"""
Gymnasium-compatible multi-agent energy environment.

Replaces 'gym' with 'gymnasium' and fixes Box dtype/bounds warnings.
Improves line-overload calculation and prints clearer per-step diagnostics.

Run:
    python energy_env_with_grid_gymnasium.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# -------------------------
# Helper physics functions
# -------------------------
def transmission_loss(power_kw, alpha=1e-4, beta=1e-6):
    p = np.asarray(power_kw, dtype=float)
    return alpha * np.abs(p) + beta * (p ** 2)


def battery_deg_cost(throughput_kwh, dod, k_throughput=0.02, k_dod=0.5):
    return k_throughput * throughput_kwh + k_dod * (dod ** 2)


# -------------------------
# Environment
# -------------------------
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

        # State per agent: [demand_kw, soc_kwh, pv_kw]
        self.state = np.zeros((self.n_agents, 3), dtype=np.float32)

        # Action per agent: [battery_power_kw, grid_trade_kw]
        action_low = np.array([-self.battery_max_charge_kw, -self.per_agent_max_kw], dtype=np.float32)
        action_high = np.array([ self.battery_max_charge_kw,  self.per_agent_max_kw], dtype=np.float32)
        # Flattened action vector length = 2 * n_agents
        self.action_space = spaces.Box(low=np.tile(action_low, (self.n_agents,)),
                                       high=np.tile(action_high, (self.n_agents,)),
                                       dtype=np.float32)

        # Observations: flatten of per-agent [demand, soc, pv]
        # Avoid np.inf in Box bounds to prevent precision warning; use large finite max
        high_val = np.finfo(np.float32).max / 4.0
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([high_val, max(1.0, self.battery_capacity_kwh), high_val], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.tile(obs_low, (self.n_agents,)),
                                            high=np.tile(obs_high, (self.n_agents,)),
                                            dtype=np.float32)

        self.rng = np.random.default_rng(seed)

        # Price signals (can be made dynamic)
        self.price_buy = 0.20  # $/kWh
        self.price_sell = 0.10

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

        obs = self._get_obs()
        # Gymnasium expects: obs, info
        return obs, {}

    def _get_obs(self):
        return self.state.flatten().astype(np.float32)

    def _apply_action_limits(self, battery_kw, grid_trade_kw):
        battery_kw = np.clip(battery_kw, -self.battery_max_charge_kw, self.battery_max_charge_kw)
        grid_trade_kw = np.clip(grid_trade_kw, -self.per_agent_max_kw, self.per_agent_max_kw)
        return battery_kw, grid_trade_kw

    def step(self, action):
        """
        Gymnasium-style: returns (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (2 * self.n_agents,), f"Expected action shape {(2*self.n_agents,)}, got {action.shape}"
        action = action.reshape(self.n_agents, 2)
        battery_action = action[:, 0]
        grid_trade = action[:, 1]
        battery_action, grid_trade = self._apply_action_limits(battery_action, grid_trade)

        demand = self.state[:, 0]
        soc = self.state[:, 1].copy()
        pv = self.state[:, 2]

        # Battery charge/discharge bookkeeping
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

            # Update SOC with split efficiency
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

        # Local net to grid before any agent grid_trade
        supply_from_local_kw = pv + effective_batt_discharge_kw
        demand_from_local_kw = demand + effective_batt_charge_kw
        local_net_to_grid_kw = supply_from_local_kw - demand_from_local_kw  # >0 export, <0 import

        # Agents propose intended injection (grid_trade). We'll enforce network limits and curtail if needed.
        intended_injection_kw = grid_trade.copy()

        total_export_kw = np.sum(np.clip(intended_injection_kw, 0, None))
        total_import_kw = np.sum(np.abs(np.clip(intended_injection_kw, None, 0)))

        # Calculate overloads (positive numbers if overload exists)
        export_overload = max(0.0, total_export_kw - self.max_line_capacity_kw)
        import_overload = max(0.0, total_import_kw - self.max_line_capacity_kw)
        line_overload_kw = max(export_overload, import_overload)

        curtailment = np.zeros(self.n_agents, dtype=float)

        # If overload in exports, proportionally scale down only exporters
        if export_overload > 0 and total_export_kw > 0:
            exporters = np.where(intended_injection_kw > 0)[0]
            export_vals = intended_injection_kw[exporters]
            scale = self.max_line_capacity_kw / (total_export_kw + 1e-12)
            new_vals = export_vals * scale
            curtail = export_vals - new_vals
            intended_injection_kw[exporters] = new_vals
            curtailment[exporters] = curtail

        # If overload in imports, proportionally scale down only importers
        if import_overload > 0 and total_import_kw > 0:
            importers = np.where(intended_injection_kw < 0)[0]
            import_vals = -intended_injection_kw[importers]
            scale = self.max_line_capacity_kw / (total_import_kw + 1e-12)
            new_vals = import_vals * scale
            curtail = import_vals - new_vals
            intended_injection_kw[importers] = -new_vals
            curtailment[importers] = curtail

        # Losses and effective injection to grid
        losses_kw = transmission_loss(intended_injection_kw, alpha=self.alpha_loss, beta=self.beta_loss)
        effective_grid_injection_kw = intended_injection_kw.copy()
        exporters_mask = intended_injection_kw > 0
        importers_mask = intended_injection_kw < 0
        # For exports, grid receives injection minus losses; for imports, agent must import extra for losses (more negative).
        effective_grid_injection_kw[exporters_mask] = intended_injection_kw[exporters_mask] - losses_kw[exporters_mask]
        effective_grid_injection_kw[importers_mask] = intended_injection_kw[importers_mask] - losses_kw[importers_mask]

        # Battery degradation
        batt_deg_costs = np.array([battery_deg_cost(batt_throughput_kwh[i], batt_dod[i],
                                                    k_throughput=self.batt_k_throughput, k_dod=self.batt_k_dod)
                                   for i in range(self.n_agents)])

        # Rewards: revenue for exports, cost for imports, minus battery deg cost, minus curtailment opportunity cost
        energy_hours = self.timestep_hours
        rewards = np.zeros(self.n_agents, dtype=float)
        for i in range(self.n_agents):
            exchanged_kwh = effective_grid_injection_kw[i] * energy_hours
            if exchanged_kwh < 0:
                rewards[i] -= (-exchanged_kwh) * self.price_buy
            else:
                rewards[i] += exchanged_kwh * self.price_sell

            if curtailment[i] > 0:
                rewards[i] -= curtailment[i] * energy_hours * self.price_sell

            # Small penalty proportional to relative contribution when overload occurred (encourage avoidance)
            if line_overload_kw > 0:
                denom = np.sum(np.abs(intended_injection_kw)) + 1e-12
                contribution = abs(intended_injection_kw[i]) / denom
                rewards[i] -= contribution * (1.0 + 0.01 * line_overload_kw)  # scaled penalty

            rewards[i] -= batt_deg_costs[i]

        # Update state with small random walk
        self.state[:, 0] = np.clip(demand + self.rng.normal(0, 1.0, size=self.n_agents), 0.0, 500.0)
        self.state[:, 1] = soc
        self.state[:, 2] = np.clip(pv + self.rng.normal(0, 2.0, size=self.n_agents), 0.0, 200.0)

        self.timestep_count += 1

        info = {
            "loss_kw": losses_kw,
            "line_overload_kw": line_overload_kw,
            "export_overload_kw": export_overload,
            "import_overload_kw": import_overload,
            "curtailment_kw": curtailment,
            "battery_deg_costs": batt_deg_costs,
            "battery_throughput_kwh": batt_throughput_kwh,
            "intended_injection_kw": intended_injection_kw,
            "effective_grid_injection_kw": effective_grid_injection_kw,
        }

        obs = self._get_obs()
        # Gymnasium expects: obs, reward, terminated, truncated, info
        terminated = False
        truncated = False
        return obs, rewards.astype(np.float32), terminated, truncated, info

    def render(self):
        print(f"t={self.timestep_count} per-agent states:")
        for i in range(self.n_agents):
            print(f" A{i}: demand={self.state[i,0]:.2f}kW soc={self.state[i,1]:.2f}kWh pv={self.state[i,2]:.2f}kW")
        print("----")

if __name__ == "__main__":
    env = EnergyMarketEnv(n_agents=4, max_line_capacity_kw=200.0, per_agent_max_kw=250.0)
    obs, _ = env.reset()
    print("obs shape:", obs.shape)
    for step in range(8):
        a = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(a)
        print(f"step {step} rewards: {rewards} | line_overload_kw: {info['line_overload_kw']:.4f} | "
              f"export_overload: {info['export_overload_kw']:.4f} import_overload: {info['import_overload_kw']:.4f}")
        print(" curtailment (per agent):", info["curtailment_kw"])
