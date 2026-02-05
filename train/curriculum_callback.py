
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning Callback.
    
    Gradually increases the difficulty of the environment by scaling penalties 
    (CO2, Grid Overload) from 0.0 to 1.0 over the first N timesteps.
    
    This helps the agent learn the basic trading task (buy low, sell high) 
    before being overwhelmed by complex constraints.
    """
    
    def __init__(self, total_timesteps: int, warmup_ratio: float = 0.3, verbose: int = 0):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.warmup_steps = int(total_timesteps * warmup_ratio)
        
    def _on_step(self) -> bool:
        # Calculate progress ratio
        curr_step = self.num_timesteps
        
        if curr_step < self.warmup_steps:
            progress = curr_step / self.warmup_steps
        else:
            progress = 1.0
            
        # Access the environment
        # SB3 wraps env in several layers (DummyVecEnv -> Monitor -> TimeLimit -> Env)
        # We need to dig down to find our custom env
        
        # Unwrap recursively
        env = self.training_env
        # Training env is a VecEnv, usually DummyVecEnv
        # Iterate over all envs if vectorized
        
        # Apply curriculum to all sub-envs
        # Note: 'get_attr' is a VecEnv method to get attributes from all envs
        # BUT 'set_attr' allows setting them.
        
        # We want to set 'co2_penalty_coeff' and 'overload_multiplier'
        
        # Base values (target hard values)
        # Ideally, we should store initial/target values in the env itself or pass them here.
        # For this refactor, let's assume the env was initialized with hard values, 
        # and we scale a 'curriculum_factor' inside the env, or we manually set the coeff.
        
        # Let's set a 'difficulty_scalar' 0.0 -> 1.0
        
        # However, EnergyMarketEnvRobust might not have 'difficulty_scalar'.
        # We can implement it, or just scale the coeffs directly if we know the targets.
        
        # Let's assume target co2=1.0, overload=50.0 (defaults)
        # We will scale them: current = target * progress
        
        new_co2 = 1.0 * progress
        new_overload = 50.0 * progress
        
        # Efficiently set attributes on all parallel envs
        self.training_env.set_attr("co2_penalty_coeff", new_co2)
        self.training_env.set_attr("overload_multiplier", new_overload)
        
        if self.verbose > 0 and curr_step % 1000 == 0:
            print(f"[Curriculum] Step {curr_step}: Difficulty {progress:.2f} (CO2={new_co2:.2f}, Overload={new_overload:.1f})")

        return True
