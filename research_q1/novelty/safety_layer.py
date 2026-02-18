
import numpy as np

class SafetyFilter:
    """
    Layer 2: Safety Filter
    Projects desired actions into the feasible set defined by physical constraints.
    """
    def __init__(self, 
                 battery_capacity_kwh: float,
                 max_power_kw: float,
                 efficiency: float = 0.9):
        
        self.capacity = battery_capacity_kwh
        self.max_power = max_power_kw
        self.efficiency = efficiency
        self.eff_sqrt = efficiency ** 0.5
        
    def get_dynamic_capacity(self, hour: int) -> float:
        """
        Returns capacity based on time (e.g., EV constraints).
        Can be overridden by subclasses or logic.
        """
        return self.capacity

    def project_action(self, 
                      current_soc: float, 
                      desired_action_kw: float,
                      dt_hours: float = 1.0,
                      current_capacity: float = None) -> float:
        """
        Clips the desired battery action to be physically feasible.
        
        Args:
            current_soc: Current State of Charge (kWh)
            desired_action_kw: Positive = Charge, Negative = Discharge
            dt_hours: Time step duration
            current_capacity: Optional dynamic max capacity (e.g. for EV)
            
        Returns:
            feasible_action_kw
        """
        if current_capacity is None:
            current_capacity = self.capacity
            
        # 1. Clip SoC to current capacity (Handling EV return)
        # If vehicle was away and returns, SoC might be > new capacity? 
        # No, if away capacity is small. If returns capacity is big.
        # But if capacity shrinks (Day time), we must clamp SoC virtually?
        # Physical battery doesn't lose energy, but accessible energy might.
        # Let's assume passed SoC is valid.
        
        # 2. Calculate Limits
        # Max Charge (Limited by space to full)
        space_kwh = max(0.0, current_capacity - current_soc)
        max_in_kw = space_kwh / dt_hours / self.eff_sqrt
        # Also limited by Inverter rating
        max_in_kw = min(max_in_kw, self.max_power)
        
        # Max Discharge (Limited by available energy)
        energy_kwh = max(0.0, current_soc) # normalized to 0 min?
        max_out_kw = energy_kwh / dt_hours * self.eff_sqrt
        # Also limited by Inverter rating
        max_out_kw = min(max_out_kw, self.max_power)
        
        # 3. Project
        # We clamp the desired action between [-max_out, max_in]
        feasible_action = np.clip(desired_action_kw, -max_out_kw, max_in_kw)
        
        return float(feasible_action)
