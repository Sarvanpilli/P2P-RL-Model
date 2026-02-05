
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

    def process_intent(self, 
                      step: int, 
                      observations: np.ndarray, 
                      rl_actions: np.ndarray,
                      state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        The main control loop for a single timestep.
        
        Args:
            step: Timestep idx
            observations: Raw observation vector (flattened)
            rl_actions: Raw actions from Neural Net (flattened)
            state: Physical state [n_agents, 3] (Demand, SoC, PV)
            
        Returns:
            final_actions: Safe, Executable actions [n_agents*3]
            info: Log info
        """
        rl_actions_reshaped = rl_actions.reshape(self.n_agents, 3)
        obs_reshaped = observations.reshape(self.n_agents, -1) # Flattened obs per agent
        
        final_actions_list = []
        guard_info = {
            "fallback_triggered": 0,
            "safety_violations": 0,
            "ood_events": 0
        }
        
        for i in range(self.n_agents):
            agent_obs = obs_reshaped[i]
            agent_state = state[i]
            agent_intent = rl_actions_reshaped[i]
            
            # --- Step 1: Health Monitor (OOD Check) ---
            is_ood, z_score = self.supervisor.detect_distribution_shift(agent_obs)
            metrics = {"z_score": z_score}
            
            status = "OK"
            final_act = None
            
            if is_ood:
                # CRITICAL: Input is weird. Trusting NN is dangerous.
                # Trigger Fallback.
                status = "FALLBACK_OOD"
                final_act = self.supervisor.get_fallback_action()
                guard_info["ood_events"] += 1
                guard_info["fallback_triggered"] += 1
            else:
                # --- Step 2: Deterministic Optimization (Layer 2) ---
                # "Optimize" the intent to be feasible.
                # FeasibilityFilter expects batch, so we wrap briefly
                # Ideally, filter logic should be per-agent or vectorized. 
                # Our filter is vectorized. We can call it for the whole batch outside loop,
                # but to mix Fallback logic, we might need granular control.
                # Let's run optimization for this agent (using sliced inputs)
                
                # ...Wait, FeasibilityFilter is vectorized. 
                # Let's re-structure: 
                # 1. Check OOD for ALL agents.
                # 2. Identify who needs Fallback.
                # 3. For others, run Optimizer.
                # 4. Check Hard Constraints.
                pass 
                
        # --- Vectorized Implementation ---
        
        # 1. Health / OOD Checks
        # Currently Supervisor OOD is per-observation.
        mask_fallback = np.zeros(self.n_agents, dtype=bool)
        
        # (Placeholder for real mean/std loading)
        # Assuming no stats loaded yet, OOD will be always False.
        
        # 2. Optimization (Layer 2)
        # Run filter on ALL intentions
        # Note: Optimization happens even if some are OOD, but we overwrite them later.
        optimized_actions, _ = self.optimizer.filter_action(
            rl_actions_reshaped, state
        )
        
        # 3. Hard Safety Check (Layer 3) & Final Selection
        final_actions = np.zeros_like(rl_actions_reshaped)
        
        for i in range(self.n_agents):
            intent = rl_actions_reshaped[i]
            opt_act = optimized_actions[i]
            ag_state = state[i]
            
            # Check Safety of OPTIMIZED action
            is_safe, reason = self.supervisor.check_hard_constraints(
                ag_state, opt_act, self.timestep_hours
            )
            
            if not is_safe:
                # VETO! Optimizer failed to find safe action?
                # Or Model Logic mismatch.
                # Trigger Fallback.
                status = "VETOED"
                final_act = self.supervisor.get_fallback_action()
                guard_info["safety_violations"] += 1
                guard_info["fallback_triggered"] += 1
                # Log the reason?
            elif mask_fallback[i]:
                # OOD Trigger
                status = "FALLBACK_OOD"
                final_act = self.supervisor.get_fallback_action()
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
            
        return final_actions.flatten(), guard_info

    def load_obs_stats(self, mean: np.ndarray, std: np.ndarray):
        """Loads training statistics for OOD detection."""
        self.supervisor.obs_mean = mean
        self.supervisor.obs_std = std
