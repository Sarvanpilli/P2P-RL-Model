
import numpy as np
from typing import Tuple, Dict, Any

from train.safety_supervisor import SafetySupervisor
from train.action_tracer import ActionTracer
from train.safety_filter import FeasibilityFilter

class AutonomousGuard:
    """
    AutonomousGuard: 3-Layer Projection-Based Safety Controller.

    Architecture:
        Layer 1 — Jitter Clipping: rate-limits action changes via slew bounds.
        Layer 2 — FeasibilityFilter: projects actions onto the physically feasible set.
                  Enforces SoC bounds, surplus-limited trading, and price validity.
        Layer 3 — SafetySupervisor: hard veto with zero-action fallback.

    Role in the hybrid safety system:
        This guard provides GUARANTEED constraint satisfaction at every timestep
        via deterministic projection. It operates as the hard backstop.

        The LagrangianSafetyLayer (train/lagrangian_safety.py) operates in parallel
        as a SOFT constraint teacher — it shapes the reward signal so the POLICY
        itself learns to avoid constraint boundaries, reducing how often this guard
        needs to intervene. Together they form a two-tier safety system:
            - Lagrangian: teaches the agent where the boundary is (learning signal)
            - AutonomousGuard: enforces the boundary if the agent crosses it (hard guarantee)

    Statistics tracked:
        guard_info['layer1_interventions']: times jitter clip fired
        guard_info['layer2_interventions']: times feasibility filter changed action
        guard_info['layer3_vetoes']:        times action was fully vetoed
        guard_info['constraint_violation_rate']: layer3_vetoes / total_steps
    """
    
    def __init__(self, 
                 n_agents: int,
                 battery_capacity_kwh: float = None,
                 battery_max_charge_kw: float = None,
                 timestep_hours: float = 1.0,
                 agent_specs: list = None,
                 normalization_stats_path: str = None,
                 **kwargs):
        
        self.n_agents = n_agents
        
        # Build per-agent specs: list of {capacity, max_charge}
        if agent_specs is not None:
            self.agent_specs = agent_specs
        elif battery_capacity_kwh is not None and battery_max_charge_kw is not None:
            # Backward compat: scalar -> replicate for all agents
            self.agent_specs = [
                {"capacity": battery_capacity_kwh, "max_charge": battery_max_charge_kw}
                for _ in range(n_agents)
            ]
        else:
            # Default fallback
            self.agent_specs = [
                {"capacity": 50.0, "max_charge": 25.0}
                for _ in range(n_agents)
            ]
        
        # Per-agent arrays for vectorized operations
        self.capacities = np.array([s["capacity"] for s in self.agent_specs])
        self.max_charges = np.array([s["max_charge"] for s in self.agent_specs])
        
        # --- Layer 2: Deterministic Optimizer ---
        # FeasibilityFilter now uses per-agent arrays
        self.optimizer = FeasibilityFilter(
            battery_capacity_kwh=self.capacities,
            battery_max_charge_kw=self.max_charges,
            timestep_hours=timestep_hours
        )
        
        # --- Layer 3: Safety Supervisor ---
        # SafetySupervisor uses per-agent capacities
        self.supervisor = SafetySupervisor(
            battery_capacity_kwh=self.capacities
        )
        
        self.tracer = ActionTracer()
        self.timestep_hours = timestep_hours

        self.prev_actions = np.zeros((n_agents, 3)) # For Jitter Clipping
        
        # --- Option A Compliance: Constraint Stats ---
        self.total_steps = 0
        self.layer1_interventions = 0
        self.layer2_interventions = 0
        self.layer3_vetoes = 0
        
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
        
        slew_limit = np.stack([
            self.max_charges,             # Batt: per-agent max charge rate
            self.max_charges * 2,         # Grid (looser)
            np.ones(self.n_agents)        # Price (arbitrary)
        ], axis=1)  # (N, 3)
        
        lower_bound = self.prev_actions - slew_limit
        upper_bound = self.prev_actions + slew_limit
        
        # Apply Clipping
        rl_actions_clipped = np.clip(rl_actions_reshaped, lower_bound, upper_bound)
        
        # Check if we actually clipped significantly
        if not np.allclose(rl_actions_clipped, rl_actions_reshaped, atol=1e-3):
             guard_info["jitter_events"] += 1

        # --- Option A Stats Increment ---
        self.total_steps += 1
        if guard_info["jitter_events"] > 0:
            self.layer1_interventions += 1
        
        # Calculate mask
        mask_fallback = np.zeros(self.n_agents, dtype=bool)
        
        # 2. Optimization (Layer 2)
        # Run filter on CLIPPED intentions
        optimized_actions, changed_l2 = self.optimizer.filter_action(
            rl_actions_clipped, state
        )
        if changed_l2:
            self.layer2_interventions += 1
        
        # 3. Hard Safety Check (Layer 3) & Final Selection
        final_actions = np.zeros_like(rl_actions_reshaped)
        step_has_veto = False
        
        for i in range(self.n_agents):
            intent = rl_actions_reshaped[i] # Log original intent
            opt_act = optimized_actions[i]
            ag_state = state[i]
            
            # Check Safety of OPTIMIZED action
            is_safe, reason = self.supervisor.check_hard_constraints(
                ag_state, opt_act, self.timestep_hours, agent_idx=i
            )
            
            if not is_safe:
                # VETO!
                status = "VETOED"
                final_act = self.supervisor.get_fallback_action()
                guard_info["safety_violations"] += 1
                guard_info["fallback_triggered"] += 1
                step_has_veto = True
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
        
        if step_has_veto:
            self.layer3_vetoes += 1

        # Update state persistence
        self.prev_actions = final_actions.copy()
            
        return final_actions.flatten(), guard_info

    def get_constraint_stats(self) -> Dict[str, Any]:
        """
        Returns cumulative safety intervention statistics.
        """
        violation_rate = self.layer3_vetoes / self.total_steps if self.total_steps > 0 else 0.0
        return {
          'total_steps': int(self.total_steps),
          'layer1_interventions': int(self.layer1_interventions),
          'layer2_interventions': int(self.layer2_interventions),
          'layer3_vetoes': int(self.layer3_vetoes),
          'constraint_violation_rate': float(violation_rate)
        }

    def load_obs_stats(self, mean: np.ndarray, std: np.ndarray):
        """Loads training statistics for OOD detection."""
        self.supervisor.obs_mean = mean
        self.supervisor.obs_std = std
