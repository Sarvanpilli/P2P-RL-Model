
import numpy as np
from enum import Enum
from typing import Dict, Any

class AgentType(Enum):
    PROSUMER = 1
    CONSUMER = 2

class MicrogridNode:
    """
    Represents a single prosumer node in the microgrid with:
    - Battery Storage (Physics & Degradation)
    - Load (Demand)
    - Generation (PV)
    """
    def __init__(self,
                 node_id: int,
                 battery_capacity_kwh: float,
                 battery_max_charge_kw: float,
                 battery_eff: float,
                 initial_soc: float = None,
                 agent_type: AgentType = AgentType.PROSUMER):
        
        self.node_id = node_id
        self.agent_type = agent_type
        
        # Physics Enforcement for Consumers
        if self.agent_type == AgentType.CONSUMER:
            self.battery_capacity_kwh = 0.0
            self.battery_max_charge_kw = 0.0
            self.battery_eff = 1.0
        else:
            self.battery_capacity_kwh = float(battery_capacity_kwh)
            self.battery_max_charge_kw = float(battery_max_charge_kw)
            self.battery_eff = float(battery_eff)
        
        # State
        if initial_soc is None:
            self.soc = self.battery_capacity_kwh * 0.5
        else:
            self.soc = float(initial_soc)
            
        self.soc = np.clip(self.soc, 0.0, self.battery_capacity_kwh)
        
        # Metrics tracking
        self.throughput_kwh = 0.0
        # Ramp-rate (optional): max power change per hour; None = disabled
        self.max_ramp_kw = None
        self.last_power_kw = 0.0

    def reset(self, soc: float = None):
        if soc is not None:
            self.soc = soc
        self.soc = np.clip(self.soc, 0.0, self.battery_capacity_kwh)
        self.throughput_kwh = 0.0

    def get_cycle_count(self) -> float:
        """Returns the equivalent full cycles performed."""
        if self.battery_capacity_kwh > 0:
            # Cycle = Total Throughput (Charge+Discharge) / (2 * Capacity)
            return self.throughput_kwh / (2.0 * self.battery_capacity_kwh)
        return 0.0

    def set_ramp_limit(self, max_ramp_kw: float):
        """Sets the maximum power change per hour (kW/h)."""
        self.max_ramp_kw = float(max_ramp_kw) if max_ramp_kw is not None else None
        self.last_power_kw = 0.0

    def _apply_ramp_constraint(self, desired_kw: float, dt_hours: float) -> float:
        """Clips desired power to be within ramp limits of last power."""
        if not hasattr(self, 'max_ramp_kw') or self.max_ramp_kw is None:
            return desired_kw
        
        # Ramp is kW per hour? Or per step? 
        # Usually Ramp Rate is defined as kW/min or %/min.
        # Let's assume input is kW/hour.
        max_delta = self.max_ramp_kw * dt_hours
        
        lower_bound = self.last_power_kw - max_delta
        upper_bound = self.last_power_kw + max_delta
        
        return float(np.clip(desired_kw, lower_bound, upper_bound))

    def step(self, 
             battery_action_kw: float, 
             current_demand_kw: float, 
             current_pv_kw: float,
             dt_hours: float) -> Dict[str, float]:
        """
        Executes physics for one timestep.
        
        Args:
            battery_action_kw: Positive = Charge, Negative = Discharge
            current_demand_kw: Local demand
            current_pv_kw: Local generation
            dt_hours: Timestep duration
            
        Returns:
            Dict with physics results:
            - effective_charge_kw
            - effective_discharge_kw
            - soc_final
            - net_load_kw (Load - PV + Charge - Discharge)
        """
        
        # 1. Enforce Consumer Physics (No PV, No Battery)
        if self.agent_type == AgentType.CONSUMER:
            current_pv_kw = 0.0
            battery_action_kw = 0.0
            
        # 1.5 Enforce Ramp Constraint (if enabled)
        # Net Battery Action
        battery_action_kw = self._apply_ramp_constraint(battery_action_kw, dt_hours)
        
        # Update last power for next step (Actual vs Desired?)
        # We should update it with the *Effective* power later.
        
        # 2. Battery Physics
        eff_sqrt = self.battery_eff ** 0.5
        
        # Desired
        req_charge = max(0.0, battery_action_kw)
        req_discharge = max(0.0, -battery_action_kw)
        
        effective_charge = 0.0
        effective_discharge = 0.0
        
        # Charge Logic
        if req_charge > 0:
            space_kwh = self.battery_capacity_kwh - self.soc
            # Max power limited by space and rate
            max_in_kw = min(self.battery_max_charge_kw, space_kwh / dt_hours / eff_sqrt) if self.battery_capacity_kwh > 0 else 0.0
            effective_charge = min(req_charge, max_in_kw)
            
            energy_added = effective_charge * dt_hours * eff_sqrt
            self.soc += energy_added
            self.throughput_kwh += energy_added
            
        # Discharge Logic
        if req_discharge > 0:
            # Max power limited by available energy and rate
            max_out_kw = min(self.battery_max_charge_kw, self.soc / dt_hours * eff_sqrt) if self.battery_capacity_kwh > 0 else 0.0
            effective_discharge = min(req_discharge, max_out_kw)
            
            energy_removed = effective_discharge * dt_hours / eff_sqrt
            self.soc -= energy_removed
            self.throughput_kwh += energy_removed
            
        # Safety Clip
        self.soc = np.clip(self.soc, 0.0, self.battery_capacity_kwh)
        
        # Store effective power for next ramp check
        # Charge is positive, Discharge is negative
        self.last_power_kw = effective_charge - effective_discharge
        
        # 3. Net Load Calculation
        # Net Load = Demand - PV + Battery_In - Battery_Out
        # Positive -> Needs Import
        # Negative -> Has Export
        net_load_kw = current_demand_kw - current_pv_kw + effective_charge - effective_discharge
        
        return {
            "node_id": self.node_id,
            "soc": self.soc,
            "effective_charge": effective_charge,
            "effective_discharge": effective_discharge,
            "throughput": self.throughput_kwh, 
            "throughput_delta": effective_charge * dt_hours * eff_sqrt + effective_discharge * dt_hours / eff_sqrt, 
            "net_load_kw": net_load_kw
        }
