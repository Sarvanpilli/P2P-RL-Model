
import numpy as np
from typing import Tuple, Dict, Any

class SafetySupervisor:
    """
    Layer 3: Hard Safety Supervisor.
    
    The Ultimate Authority. Enforces invariats and monitors system health.
    
    Responsibilities:
    1. Invariant Checks (SoC bounds, Energy Conservation).
    2. Distribution Shift Detection (OOD).
    3. Fallback Generation (Safe Mode).
    """
    
    def __init__(self, 
                 battery_capacity_kwh: float,
                 obs_mean: np.ndarray = None, 
                 obs_std: np.ndarray = None,
                 ood_threshold_std: float = 4.0):
        
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        
        # Health Monitor: OOD Detection params
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.ood_threshold_std = ood_threshold_std
        
        # Runtime stats
        self.consecutive_violations = 0
        self.max_violations_before_lockout = 5

    def check_hard_constraints(self, state: np.ndarray, action: np.ndarray, dt: float) -> Tuple[bool, str]:
        """
        Verifies if the proposed action satisfies physical invariants.
        
        Args:
            state: [Demand, SoC, PV]
            action: [Battery_kW, Trade_kW, Price]
            dt: Timestep hours
            
        Returns:
            passed: Bool
            reason: String
        """
        demand = state[0]
        soc = state[1]
        pv = state[2]
        
        batt_kw = action[0] # (+) Charge, (-) Discharge
        trade_kw = action[1]
        
        # Invariant 1: SoC Bounds
        # Predict next SoC
        eff = 0.95 ** 0.5 # Approx sqrt eff
        if batt_kw > 0:
            next_soc = soc + batt_kw * dt * eff
        else:
            next_soc = soc + batt_kw * dt / eff # batt_kw is negative
            
        # Tolerance 1e-3
        if next_soc < -1e-3 or next_soc > self.battery_capacity_kwh + 1e-3:
            return False, f"SoC Violation: Next {next_soc:.2f} outside [0, {self.battery_capacity_kwh}]"
            
        # Invariant 2: Energy Conservation (Local)
        # Power Balance: PV + Discharge + Import = Demand + Charge + Export + Losses
        # Simplified Check: Do we have enough resource to support Export?
        # Net_Gen = PV - Demand - Battery_Flow (Batt>0 is load)
        # Net_Gen = PV - Demand - Batt_KW
        # If Net_Gen < 0 (Deficit), we CANNOT Export (Trade > 0).
        # We MUST Import (Trade <= Net_Gen).
        
        net_resource = pv - demand - batt_kw
        
        if net_resource < -1e-5:
            # We are in deficit
            if trade_kw > 1e-5:
                 return False, f"Physics Violation: Exporting {trade_kw:.2f}kW while in deficit {net_resource:.2f}kW"
            # We must import at least expected amount? 
            # Actually, if we don't import enough, we have Loss of Load (Blackout).
            # The system might allow blackouts (penalized), but SAFETY wise?
            # Creating energy out of thin air is forbidden.
            # If Trade > Net_Resource (and both negative), e.g. Res=-5, Trade=-2.
            # Imbalance = -3. Where did it come from? 
            # In simulation, it's Load Shedding.
            # But let's assume we enforce: Trade <= Net_Resource (algebraically).
            # e.g. -10 <= -5 -> True. Import 10 covers deficit 5.
            # Wait, Import is negative. So -10 <= -5 is True? Yes.
            # So if we import MORE than needed, it's fine (curtailed/dumped).
            pass
        else:
            # We are in surplus
            # We cannot export MORE than surplus.
            if trade_kw > net_resource + 1e-5:
                return False, f"Physics Violation: Exporting {trade_kw:.2f}kW exceeds surplus {net_resource:.2f}kW"
                
        return True, "OK"

    def detect_distribution_shift(self, obs: np.ndarray) -> Tuple[bool, float]:
        """
        Checks if observation is Out-Of-Distribution (OOD).
        
        Returns:
            is_ood: Bool
            z_score_max: Max deviation magnitude
        """
        if self.obs_mean is None or self.obs_std is None:
            return False, 0.0
            
        # Standardize
        # Avoid div by zero
        std_safe = np.where(self.obs_std < 1e-6, 1.0, self.obs_std)
        z_scores = np.abs((obs - self.obs_mean) / std_safe)
        
        max_z = np.max(z_scores)
        
        if max_z > self.ood_threshold_std:
            return True, float(max_z)
            
        return False, float(max_z)

    def get_fallback_action(self) -> np.ndarray:
        """
        Emergency Safe Action.
        
        Strategy:
        1. Battery Idle (0 kW). Minimize risk of thermal/SoC violation.
        2. Trade Zero (0 kW). Disconnect from P2P market (Grid-tied Islanding).
        3. Price Zero (Passive).
        """
        # [Batt_KW, Trade_KW, Price]
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
