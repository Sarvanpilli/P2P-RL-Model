
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# CONFIG
AGENT_COUNTS = [4, 8, 12, 16]
R_PROSUMER = 0.6
RESULTS_DIR = "research_q1/results/scalability"
MODEL_DIR = "models_scalable"

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
        infos = [info_dict for _ in range(self.num_envs)]
        
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

def evaluate_scalability(model, n_agents, steps=1000):
    print(f"--- Final Evaluation N={n_agents} ---")
    eval_env = VectorizedMultiAgentEnv(n_agents=n_agents)
    
    obs = eval_env.reset()
    
    p2p_history = []
    dep_history = []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        
        # Get market metrics from info (shared across all agents in one step)
        info = infos[0]
        p2p_history.append(info.get('p2p_volume_kwh_step', 0))
        dep_history.append(info.get('rolling_grid_dependency', 1.0))
            
    eval_env.close()
    
    return {
        'Agents': n_agents,
        'Grid %': np.mean(dep_history) * 100.0,
        'P2P (kWh)': np.sum(p2p_history),
        'Success %': (np.count_nonzero(p2p_history) / steps) * 100.0
    }

def run_scalability_sweep():
    summary = []
    current_model = None
    
    for n in AGENT_COUNTS:
        print(f"\n===== STAGED TRAINING N={n} (Transfer Learning) =====")
        
        # 1. Setup Env (N agents)
        train_env = VectorizedMultiAgentEnv(n_agents=n)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        # 2. Initialize or Load
        # 2. Initialize and Transfer
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4)
        if current_model is not None:
            # Manual parameter transfer to bypass n_envs check
            params = current_model.get_parameters()
            model.set_parameters(params)
            print(f"Policy weights transferred from N to N={n} architecture.")
            
        # 3. Train (Selective: 10,000 steps per phase)
        model.learn(total_timesteps=10000)
        
        # 4. Save
        model_p = os.path.join(MODEL_DIR, f"ppo_n{n}")
        model.save(model_p)
        
        # 5. Evaluate
        res = evaluate_scalability(model, n)
        summary.append(res)
        
        current_model = model
        train_env.close()
        
    # Final Summary Table
    df = pd.DataFrame(summary)
    print("\n\n" + "="*50)
    print("SCALABILITY EXPERIMENT RESULTS")
    print("="*50)
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(RESULTS_DIR, "scalability_metrics_v2.csv"), index=False)
    return df

if __name__ == "__main__":
    run_scalability_sweep()
