import numpy as np

class FeasibilityFilter:
    """
    Enforces hard physical constraints on agent actions.
    """
    def __init__(self, 
                 battery_capacity_kwh=50.0,
                 battery_max_charge_kw=25.0,
                 timestep_hours=1.0):
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.battery_max_charge_kw = float(battery_max_charge_kw)
        self.timestep_hours = float(timestep_hours)

    def filter_action(self, raw_action, state, grid_buy_price=0.20, grid_sell_price=0.10):
        """
        Adjusts raw_action to be feasible given the current state.
        Deterministic, Stateless, Idempotent.
        
        Args:
            raw_action: np.array of shape (n_agents, 3) -> [batt_kw, grid_kw, price_bid]
            state: np.array of shape (n_agents, 3) -> [demand, soc, pv]
            grid_buy_price: float, max reasonable price
            grid_sell_price: float, min reasonable price
            
        Returns:
            safe_action: np.array of same shape
            changed: bool, True if any modification was made
        """
        safe_action = raw_action.copy()
        n_agents = safe_action.shape[0]
        
        demand = state[:, 0]
        soc = state[:, 1]
        pv = state[:, 2]
        
        # 1. Clip Battery Power (Physics)
        # Max Charge: Limited by Capacity - SoC AND Max Rate
        max_charge_kwh = np.maximum(0.0, self.battery_capacity_kwh - soc)
        max_charge_kw = np.minimum(self.battery_max_charge_kw, max_charge_kwh / self.timestep_hours)
        
        # Max Discharge: Limited by SoC AND Max Rate
        max_discharge_kwh = soc
        max_discharge_kw = np.minimum(self.battery_max_charge_kw, max_discharge_kwh / self.timestep_hours)
        
        # Apply Clipping
        # We use a loop or vector ops. Vector ops are cleaner.
        batt_kw = safe_action[:, 0]
        batt_kw = np.clip(batt_kw, -max_discharge_kw, max_charge_kw)
        safe_action[:, 0] = batt_kw
        
        # 2. Clip Grid Trade (Energy Balance)
        # Surplus = PV + Discharge - Charge - Demand
        # If Surplus > 0: Can Export (positive trade) up to Surplus
        # If Surplus < 0: Must Import (negative trade) at least Deficit
        
        # Note: We treat battery action as "committed" for this step.
        # Discharge is negative in action, but adds to supply.
        # Charge is positive in action, adds to load.
        
        # Net Generation = PV
        # Net Load = Demand
        # Battery Flow = batt_kw (positive = load, negative = generation)
        
        # Available for Export = PV - Demand - batt_kw
        # If > 0: Max Export = val
        # If < 0: Max Import = val (negative)
        
        # Wait, strictly speaking:
        # If we have surplus, we CAN export, but we don't HAVE to (curtailment).
        # If we have deficit, we MUST import (or load shed, but we treat load as hard).
        
        # Let's define:
        # Net_Position = PV - Demand - batt_kw
        
        net_position = pv - demand - batt_kw
        
        trade_kw = safe_action[:, 1]
        
        for i in range(n_agents):
            pos = net_position[i]
            if pos >= 0:
                # We have surplus 'pos'.
                # Can export up to 'pos'.
                # Can import? No, why would we? But technically we could charge more?
                # No, battery is already fixed.
                # So trade must be <= pos.
                # Can it be negative? (Importing while having surplus).
                # Yes, if we want to dump power? No, that's wasteful.
                # Let's restrict: trade must be <= pos.
                # And usually >= -max_line (but we assume we don't import if we have surplus).
                # Let's just clip max export.
                trade_kw[i] = min(trade_kw[i], pos)
                
                # Also, we shouldn't import if we have surplus, unless we really want to.
                # But for safety, let's say we can't export MORE than we have.
                pass
            else:
                # We have deficit 'pos' (negative).
                # We need to import at least 'abs(pos)'.
                # So trade must be <= pos (since import is negative).
                # Wait, if pos is -5, we need -5 or less (e.g. -10).
                # So trade must be <= pos.
                # trade_kw[i] = min(trade_kw[i], pos) # This forces import.
                
                # Actually, the environment usually handles the "gap" by forcing grid.
                # But the filter should ensure the agent *intends* to cover it?
                # The audit says: "Clip trade to available surplus/deficit".
                # If I have 5kW surplus, I can't sell 10kW.
                # If I have 5kW deficit, I can't sell ANYTHING. I must buy.
                
                # Let's enforce:
                # Max Export = max(0, pos)
                # Min Import = -infinity (or line limit)
                
                # If pos < 0 (Deficit):
                # Max Export = 0.
                # Agent cannot sell.
                trade_kw[i] = min(trade_kw[i], max(0.0, pos))
                
        safe_action[:, 1] = trade_kw
        
        # 3. Clip Price Bid
        # Reasonable bounds: [0.0, 1.0] or [Feed-in, Retail]
        # Audit suggests: [feed_in, retail]
        # Let's be slightly generous [0.0, 1.0] to allow learning, but strict is better for stability.
        # Let's use [0.0, 1.0] as hard bounds.
        safe_action[:, 2] = np.clip(safe_action[:, 2], 0.0, 1.0)
        
        # Check for changes
        # Use a small tolerance for float comparison
        changed = not np.allclose(raw_action, safe_action, atol=1e-5)
        
        return safe_action, changed
