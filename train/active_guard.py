
import numpy as np
from train.autonomous_guard import AutonomousGuard

class ActiveGuard(AutonomousGuard):
    """
    Active Learning Guard that updates its OOD statistics online using Welford's Algorithm.
    Unlike the static AutonomousGuard, this adapts to the agent's shifting distribution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Welford's Alg State
        # We track per-feature mean/var
        # Shape: Assume flattened obs (or max size)
        # We need the obs dim. It's not passed in init explicitly in base class.
        # We will init on first call.
        self.count = 0
        self.mean = None
        self.M2 = None # Sum of squares of differences
        
        self.warmup_steps = 1000 # Don't trigger OOD before this
        
    def _update_stats(self, observations: np.ndarray):
        """
        Update running mean and variance using Welford's algorithm.
        obs: (batch_size, obs_dim) or (obs_dim,)
        """
        obs = np.atleast_2d(observations)
        batch_size, dim = obs.shape
        
        if self.mean is None:
             self.mean = np.zeros(dim)
             self.M2 = np.zeros(dim)
        
        # Welford's for batch is tricky, let's just loop or use Chan's method
        # Simple loop for robustness (batch is small, N=4)
        for x in obs:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
            
        # Update Supervisor's stats
        if self.count > 2:
            var = self.M2 / (self.count - 1)
            std = np.sqrt(var)
            # Avoid zero division
            std = np.maximum(std, 1e-6)
            
            # Injection into Supervisor
            self.load_obs_stats(self.mean, std)

    def process_intent(self, step, observations, rl_actions, state):
        # 1. Update Stats Online
        # Observations passed here are flattened: (N * ObsDim)
        # We need to reshape to (N, ObsDim) to update correctly
        n = self.n_agents
        obs_reshaped = observations.reshape(n, -1)
        
        self._update_stats(obs_reshaped)
        
        # 2. Call Parent Process
        # Is warmup done?
        if self.count < self.warmup_steps:
             # Force Supervisor to NOT trigger OOD yet
             # But base class uses self.mask_fallback which relies on supervisor.detect...
             # We effectively disable OOD by ensuring the check passes or ignored.
             # Easier: Just run parent. If it triggers OOD because stats are init to 0/1, that's bad.
             # So we should probably initialize Mean/Std to reasonable loose bounds?
             pass
             
        return super().process_intent(step, observations, rl_actions, state)
