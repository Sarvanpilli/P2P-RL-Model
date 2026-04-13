
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# CONFIG
AGENT_COUNTS = [4, 8, 12, 16]
R_PROSUMER = 0.6
RESULTS_DIR = "research_q1/results/scalability_v3"
MODEL_DIR = "models_scalable_v3"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class VectorizedMultiAgentEnv(VecEnv):
    """
    Transparently presents a single multi-agent EnergyMarketEnv Robust 
    as a Vectorized environment for SB3, where each agent is an independent 'env'.
    """
    def __init__(self, n_agents=4):
        n_pro = int(n_agents * R_PROSUMER)
        n_con = n_agents - n_pro
        self.env = EnergyMarketEnvRobust(n_prosumers=n_pro, n_consumers=n_con)
        self.num_envs = n_agents
        
        # Spaces are per-agent
        super().__init__(
            num_envs=n_agents,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        self.last_obs = None

    def reset(self):
        obs, info = self.env.reset()
        self.last_obs = obs # (N, obs_dim)
        return self.last_obs

    def step_async(self, actions):
        self.actions = actions # (N, 3)

    def step_wait(self):
        # Step the underlying microgrid
        obs, rewards, done, truncated, info_dict = self.env.step(self.actions)
        
        # SB3 expects rewards as (N,)
        # done/truncated as (N,)
        dones = np.array([done or truncated] * self.num_envs)
        
        # Info: SB3 expects a list of dicts, one per env
        # Step 6: Diagnostic Logging
        infos = []
        for i in range(self.num_envs):
            inf = info_dict.copy()
            # Agent-specific info if needed
            infos.append(inf)
        
        if done or truncated:
            obs, _ = self.env.reset()
            
        self.last_obs = obs
        return self.last_obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def seed(self, seed=None):
        return [self.env.reset(seed=seed) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

class JointEarlyStoppingCallback(BaseCallback):
    """
    Step 7: Stop if no improvement in trade_success_rate AND grid_dependency for 50k steps.
    """
    def __init__(self, check_freq: int = 2048, patience_steps: int = 50000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience_steps = patience_steps
        self.best_success = -float('inf')
        self.best_grid_dep = float('inf')
        self.steps_without_improvement = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Stats are shared in the microgrid info
            info = self.locals['infos'][0]
            success = info.get('last_success_rate', 0.0)
            grid_dep = info.get('rolling_grid_dependency', 1.0)
            
            improved = False
            if success > self.best_success + 1e-4:
                self.best_success = success
                improved = True
            
            if grid_dep < self.best_grid_dep - 1e-4:
                self.best_grid_dep = grid_dep
                improved = True
                
            if improved:
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += self.check_freq
                
            if self.verbose > 0:
                print(f"[Callback] Steps: {self.num_timesteps}, Success: {success:.3f}, GridDep: {grid_dep:.3f}, Idle: {self.steps_without_improvement}")
                
            if self.steps_without_improvement >= self.patience_steps:
                if self.verbose > 0:
                    print(f"Early stopping at step {self.num_timesteps} due to joint metric plateau.")
                return False
        return True

def evaluate_scalability(model, n_agents, steps=2000):
    print(f"--- Final Evaluation N={n_agents} ---")
    eval_env = VectorizedMultiAgentEnv(n_agents=n_agents)
    
    obs = eval_env.reset()
    
    p2p_history = []
    dep_history = []
    success_history = []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        
        info = infos[0]
        p2p_history.append(info.get('p2p_volume_kwh_step', 0))
        dep_history.append(info.get('rolling_grid_dependency', 1.0))
        success_history.append(float(info.get('p2p_volume_kwh_step', 0) > 1e-3))
            
    eval_env.close()
    
    return {
        'Agents': n_agents,
        'Grid %': np.mean(dep_history) * 100.0,
        'P2P (kWh)': np.sum(p2p_history),
        'Success %': (np.mean(success_history)) * 100.0
    }

def run_scalability_sweep():
    summary = []
    current_model = None
    
    # Step 8: Controlled Staged Training (100k -> 200k -> 300k)
    # Re-interpreting: N=4 (100k), N=8 (200k), N=12 (300k), N=16 (300k)
    training_steps = {4: 100000, 8: 200000, 12: 300000, 16: 300000}
    
    for n in AGENT_COUNTS:
        print(f"\n===== STAGED TRAINING N={n} (Transfer Learning) =====")
        
        train_env = VectorizedMultiAgentEnv(n_agents=n)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4)
        if current_model is not None:
            params = current_model.get_parameters()
            model.set_parameters(params)
            print(f"Policy weights transferred from N to N={n} architecture.")
            
        # Step 7: Early Stopping
        callback = JointEarlyStoppingCallback(patience_steps=50000)
        
        total_steps = training_steps[n]
        model.learn(total_timesteps=total_steps, callback=callback)
        
        model_p = os.path.join(MODEL_DIR, f"ppo_n{n}")
        model.save(model_p)
        train_env.save(os.path.join(MODEL_DIR, f"vec_n{n}.pkl"))
        
        res = evaluate_scalability(model, n)
        summary.append(res)
        
        current_model = model
        train_env.close()
        
    # Final Table
    df = pd.DataFrame(summary)
    print("\n\n" + "="*50)
    print("SCALABILITY EXPERIMENT RESULTS (v3)")
    print("="*50)
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(RESULTS_DIR, "scalability_metrics_v3.csv"), index=False)
    return df

if __name__ == "__main__":
    run_scalability_sweep()
