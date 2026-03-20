import numpy as np
from typing import Dict, List, Any

"""
Lagrangian Primal-Dual Safety Layer for P2P Energy Trading MARL.

This module implements soft constraint enforcement via Lagrangian relaxation,
complementing the hard-clipping architecture in autonomous_guard.py.

Two-tier safety architecture:
    1. LagrangianSafetyLayer (this file):
       - Adds a differentiable penalty to the PPO reward
       - Updates Lagrange multipliers lambda_k after each episode
       - Teaches the neural network to stay away from constraint boundaries
       - Does NOT modify actions — shapes the learning signal instead
       - Theoretical guarantee: if lambda converges, the policy satisfies
         the constraints in expectation at the specified threshold level

    2. AutonomousGuard (autonomous_guard.py):
       - Hard projects actions onto feasible set
       - Guarantees constraint satisfaction at EVERY step regardless of policy
       - Operates as backstop when Lagrangian has not yet converged

Constraints enforced:
    C1: Battery SoC in [0, battery_capacity_kwh] for all agents
    C2: Total line flow <= max_line_capacity_kw
    C3: Voltage deviation <= 5% of nominal grid voltage

References:
    Achiam et al. (2017), "Constrained Policy Optimization", ICML
    Tessler et al. (2018), "Reward Constrained Policy Optimization", ICLR
"""

class LagrangianSafetyLayer:
    """
    Primal-Dual Lagrangian safety constraint enforcement for P2P MARL.

    Unlike hard-clipping (FeasibilityFilter), this layer:
    - Does NOT modify actions before execution
    - Computes a penalty term added to the PPO reward signal
    - Updates Lagrange multipliers (lambda) after each episode via gradient ascent
    - Forces the POLICY ITSELF to learn constraint satisfaction
    - The FeasibilityFilter remains as a hard backstop for safety

    Theory:
      Constrained RL objective:
        max E[R_t]  subject to E[C_k] <= threshold_k for k in {1,2,3}

      Lagrangian relaxation:
        L = E[R_t] - sum_k(lambda_k * max(0, C_k - threshold_k))

      Dual update (gradient ascent on lambda):
        lambda_k <- max(0, lambda_k + alpha * (violation_k - threshold_k))
    """

    def __init__(
        self,
        n_agents: int = 4,
        alpha: float = 0.005,          # dual learning rate
        threshold_soc_violation: float = 0.01,   # allow 1% avg SoC violation
        threshold_line_violation: float = 0.05,  # allow 5% line capacity overage
        threshold_voltage_violation: float = 0.03, # allow 3% voltage deviation
        max_lambda: float = 10.0,      # cap lambdas to prevent explosion
        lambda_init: float = 0.1,      # warm start (not zero, avoids cold start)
    ):
        self.n_agents = n_agents
        self.alpha = alpha
        self.thresholds = np.array([
            threshold_soc_violation,
            threshold_line_violation,
            threshold_voltage_violation
        ], dtype=np.float32)
        self.max_lambda = max_lambda

        # Lagrange multipliers — one per constraint
        # Shape: (3,) → [lambda_soc, lambda_line, lambda_voltage]
        self.lambdas = np.full(3, lambda_init, dtype=np.float32)

        # Episode-level accumulators (reset each episode)
        self._episode_violations = []   # list of (3,) arrays per step
        self._step_count = 0

        # Lifetime statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.lambda_history = []        # list of (3,) arrays, one per episode end

    def compute_violations(
        self,
        soc_values: np.ndarray,           # shape (n_agents,), in kWh
        battery_capacities: np.ndarray,   # shape (n_agents,), in kWh
        line_flow_kw: float,              # scalar: total abs network flow
        max_line_capacity_kw: float,      # scalar: physical line limit
        grid_voltage_kv: float = 0.4,     # nominal voltage
        line_resistance_ohm: float = 0.01 # line resistance
    ) -> np.ndarray:
        """
        Compute normalized constraint violations at the current timestep.

        Returns:
            violations: np.ndarray shape (3,)
                [soc_violation, line_violation, voltage_violation]
                Each value is 0.0 if constraint satisfied, >0 if violated.
        """
        # C1: SoC violation
        # Violation if any SoC < 0 or SoC > capacity
        soc_upper_breach = np.maximum(0.0, soc_values - battery_capacities)
        soc_lower_breach = np.maximum(0.0, -soc_values)
        soc_violation = float(np.mean(soc_upper_breach + soc_lower_breach))

        # Normalize by mean capacity to make scale-independent
        mean_capacity = np.mean(battery_capacities)
        if mean_capacity > 1e-9:
            soc_violation /= mean_capacity

        # C2: Line flow violation
        # Violation if total network flow exceeds physical limit
        if max_line_capacity_kw > 1e-9:
            line_violation = float(
                max(0.0, line_flow_kw - max_line_capacity_kw) / max_line_capacity_kw
            )
        else:
            line_violation = 0.0

        # C3: Voltage deviation violation
        # Simplified: estimate voltage drop from I*R = (P/V)*R
        # Violation if deviation > 5% of nominal
        if grid_voltage_kv > 1e-9:
            current_amps = line_flow_kw / grid_voltage_kv
            voltage_drop_kv = current_amps * line_resistance_ohm / 1000.0
            voltage_deviation = voltage_drop_kv / grid_voltage_kv
            voltage_violation = float(max(0.0, voltage_deviation - 0.05))
        else:
            voltage_violation = 0.0

        return np.array([soc_violation, line_violation, voltage_violation],
                        dtype=np.float32)

    def compute_penalty(self, violations: np.ndarray) -> float:
        """
        Compute the Lagrangian penalty term to SUBTRACT from the reward.

        penalty = sum_k(lambda_k * max(0, violation_k - threshold_k))

        This is added to the PPO reward as a NEGATIVE term.
        The agent experiences heavier penalties as lambda grows.
        """
        excess_violations = np.maximum(0.0, violations - self.thresholds)
        penalty = float(np.dot(self.lambdas, excess_violations))
        return penalty

    def record_step(self, violations: np.ndarray):
        """
        Record violations for this timestep. Call once per env.step().
        """
        self._episode_violations.append(violations.copy())
        self._step_count += 1
        self.total_steps += 1

    def end_episode_update(self):
        """
        Perform the dual update at the end of each episode.
        Called inside env.reset() before clearing episode state.

        Dual update rule:
            lambda_k <- clip(lambda_k + alpha * (mean_violation_k - threshold_k), 0, max_lambda)

        If violations > threshold: lambda increases → heavier penalty next episode
        If violations < threshold: lambda decreases → lighter penalty next episode
        """
        if len(self._episode_violations) == 0:
            return

        # Mean violation across all steps in this episode
        mean_violations = np.mean(self._episode_violations, axis=0)

        # Gradient ascent on lambda
        self.lambdas += self.alpha * (mean_violations - self.thresholds)

        # Enforce non-negativity and cap
        self.lambdas = np.clip(self.lambdas, 0.0, self.max_lambda)

        # Log for TensorBoard
        self.lambda_history.append(self.lambdas.copy())
        self.total_episodes += 1

        # Reset episode accumulators
        self._episode_violations = []
        self._step_count = 0

    def get_lambda_info(self) -> dict:
        """
        Return current lambda values for logging into info dict.
        """
        return {
            'lagrangian/lambda_soc':     float(self.lambdas[0]),
            'lagrangian/lambda_line':    float(self.lambdas[1]),
            'lagrangian/lambda_voltage': float(self.lambdas[2]),
            'lagrangian/total_penalty':  float(np.sum(self.lambdas)),
            'lagrangian/total_episodes': self.total_episodes,
        }

    def get_stats_summary(self) -> str:
        """
        Human-readable summary. Call at end of training to verify convergence.
        """
        if len(self.lambda_history) == 0:
            return "LagrangianSafetyLayer: no episodes completed yet."

        final_lambdas = self.lambda_history[-1]
        
        # Convergence check: last 5 iterations stable
        if len(self.lambda_history) >= 6:
            recent = np.array(self.lambda_history[-6:])
            diffs = np.abs(np.diff(recent, axis=0))
            converged = np.all(diffs < 0.001)
        else:
            converged = False

        return (
            f"LagrangianSafetyLayer Summary:\n"
            f"  Episodes:            {self.total_episodes}\n"
            f"  Total steps:         {self.total_steps}\n"
            f"  Final lambda_soc:    {final_lambdas[0]:.4f}\n"
            f"  Final lambda_line:   {final_lambdas[1]:.4f}\n"
            f"  Final lambda_volt:   {final_lambdas[2]:.4f}\n"
            f"  Converged (last 5):  {converged}\n"
            f"  Thresholds:          {self.thresholds}"
        )
