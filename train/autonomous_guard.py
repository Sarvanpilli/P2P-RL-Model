
import numpy as np
from typing import Tuple, Dict, Any

from train.safety_supervisor import SafetySupervisor
from train.action_tracer import ActionTracer
from train.safety_filter import FeasibilityFilter

class AutonomousGuard:
    """
    The Central Nervous System of the Autonomous Controller.
    
    Implements the 3-Layer Architecture:
    1. Layer 1 (Input): Strategic RL Intent.
    2. Layer 2 (Process): Deterministic Optimization (via FeasibilityFilter).
    3. Layer 3 (Verify): Hard Safety Supervisor & Health Monitor.
    
    Also handles:
    - Tracing/Logging
    - OOD Detection -> Fallback
    """
    
    def __init__(self, 
                 n_agents: int,
                 battery_capacity_kwh: float,
                 battery_max_charge_kw: float,
                 timestep_hours: float,
                 normalization_stats_path: str = None):
        
        self.n_agents = n_agents
        
        # --- Layer 2: Deterministic Optimizer ---
        # We reuse the existing FeasibilityFilter as our "Optimizer"
        # because it projects actions into the valid set (clipping).
        self.optimizer = FeasibilityFilter(
            battery_capacity_kwh=battery_capacity_kwh,
            battery_max_charge_kw=battery_max_charge_kw,
            timestep_hours=timestep_hours
        )
        
        # --- Layer 3: Safety Supervisor ---
        self.supervisor = SafetySupervisor(
            battery_capacity_kwh=battery_capacity_kwh
            # Todo: Load obs stats for OOD detection
        )
        
        self.tracer = ActionTracer()
        self.timestep_hours = timestep_hours

        self.prev_actions = np.zeros((n_agents, 3)) # For Jitter Clipping
        
    def reset(self):
        """Resets internal state."""
        self.prev_actions.fill(0.0)

    def process_intent(self, 
                      step: int, 
                      observations: np.ndarray, 
                      rl_actions: np.ndarray,
                      state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        The main control loop for a single timestep.
        """
        rl_actions_reshaped = rl_actions.reshape(self.n_agents, 3)
        obs_reshaped = observations.reshape(self.n_agents, -1) # Flattened obs per agent
        
        final_actions_list = []
        guard_info = {
            "fallback_triggered": 0,
            "safety_violations": 0,
            "ood_events": 0,
            "jitter_events": 0
        }
        
        # --- Jitter Clipping (Pre-Optim) ---
        # Limit the rate of change of action to prevent high-frequency oscillations.
        # Max Slew Rate: 0.8 (allows 80% range swing per step).
        # This prevents -1.0 to 1.0 (2.0 swing) in one step, but allows agile movement.
        max_slew = 0.8
        
        # Clip current intent to be within [prev - slew, prev + slew]
        # Actions are roughly expected to be in [-1, 1] or scaled. 
        # But wait, rl_actions might be large if unnormalized?
        # The env action space is physical kW.
        # Oh, strict normalization mentioned in requirements. 
        # If env uses normalized PPO actions [-1, 1], then this works.
        # But 'raw_action' comes from Env.step -> is it physical or normalized?
        # PPO outputs [-1, 1]. Env wrapper scales it?
        # EnergyMarketEnvRobust defines action_space as Box(-max, max).
        # So PPO unscales it? Or we are using VecNormalize?
        # We are using VecNormalize with norm_obs=False, norm_reward=True.
        # But standard PPO assumes action space is target. SB3 will unscale if it knows bounds? No.
        # PPO outputs clipped to action space. 
        # So inputs here are PHYSICAL kW.
        # If physical, 0.8 kW is tiny. 
        # We need relative slew. 0.8 * Capacity?
        # Let's assume normalized actions inside Guard?
        # "rl_actions" passed to Guard are from "raw_action" in Env.py.
        # env.step(action): raw_action = action.reshape...
        # So it IS physical.
        
        # Let's derive max slew from capacity.
        # Max Charge = battery_max_charge_kw. 
        # Allow full range in 2 steps? -> Max Slew = Max Charge.
        # Let's use 100% of Max Charge as limit per step.
        # Ideally, we allow 0 -> Max in one step. But -Max -> Max in one step is bad.
        # Let's limit change to 1.5 * limit ? 
        # Or just skip this complicated check and rely on REWARD penalty?
        # User REQ: "Clip any jittery actions".
        # Let's implement a safe slew rate of "Max Charge kW" per step.
        # If I was at -25kW, I can go to 0kW, but not +25kW.
        
        slew_limit = np.array([
            self.optimizer.battery_max_charge_kw, # Batt
            self.optimizer.battery_max_charge_kw * 2, # Grid (looser)
            1.0 # Price (arbitrary)
        ])
        
        lower_bound = self.prev_actions - slew_limit
        upper_bound = self.prev_actions + slew_limit
        
        # Apply Clipping
        rl_actions_clipped = np.clip(rl_actions_reshaped, lower_bound, upper_bound)
        
        # Check if we actually clipped significantly
        if not np.allclose(rl_actions_clipped, rl_actions_reshaped, atol=1e-3):
             guard_info["jitter_events"] += 1

        # Calculate mask
        mask_fallback = np.zeros(self.n_agents, dtype=bool)
        
        # 2. Optimization (Layer 2)
        # Run filter on CLIPPED intentions
        optimized_actions, _ = self.optimizer.filter_action(
            rl_actions_clipped, state
        )
        
        # 3. Hard Safety Check (Layer 3) & Final Selection
        final_actions = np.zeros_like(rl_actions_reshaped)
        
        for i in range(self.n_agents):
            intent = rl_actions_reshaped[i] # Log original intent
            opt_act = optimized_actions[i]
            ag_state = state[i]
            
            # Check Safety of OPTIMIZED action
            is_safe, reason = self.supervisor.check_hard_constraints(
                ag_state, opt_act, self.timestep_hours
            )
            
            if not is_safe:
                # VETO!
                status = "VETOED"
                final_act = self.supervisor.get_fallback_action()
                guard_info["safety_violations"] += 1
                guard_info["fallback_triggered"] += 1
            else:
                # All Good
                status = "OPTIMIZED"
                final_act = opt_act
                
            final_actions[i] = final_act
            
            # 4. Trace
            self.tracer.log_decision(
                step=step,
                agent_id=i,
                obs=obs_reshaped[i],
                rl_intent=intent,
                optimized_action=opt_act,
                final_action=final_act,
                safety_status=status,
                metrics={"reason": reason if not is_safe else "OK"}
            )
        
        # Update state persistence
        self.prev_actions = final_actions.copy()
            
        return final_actions.flatten(), guard_info

    def load_obs_stats(self, mean: np.ndarray, std: np.ndarray):
        """Loads training statistics for OOD detection."""
        self.supervisor.obs_mean = mean
        self.supervisor.obs_std = std
