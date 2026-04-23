
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# CONFIG
AGENT_COUNTS = [4] 
R_PROSUMER = 0.6
RESULTS_DIR = "research_q1/results/v7_emergence"
MODEL_DIR = "models_v7_emergence"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class VectorizedMultiAgentEnv(VecEnv):
    def __init__(self, n_agents=4, use_alignment_reward=True, use_curriculum=True):
        n_pro = int(n_agents * R_PROSUMER)
        n_con = n_agents - n_pro
        self.env = EnergyMarketEnvRobust(
            n_prosumers=n_pro, 
            n_consumers=n_con,
            use_alignment_reward=use_alignment_reward,
            use_curriculum=use_curriculum
        )
        self.num_envs = n_agents
        
        super().__init__(
            num_envs=n_agents,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        self.last_obs = None

    def reset(self):
        obs, info = self.env.reset()
        self.last_obs = obs
        return self.last_obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, done, truncated, info_dict = self.env.step(self.actions)
        if np.isscalar(rewards):
            rewards = np.array([rewards] * self.num_envs, dtype=np.float32)
            
        dones = np.array([done or truncated] * self.num_envs)
        infos = [info_dict.copy() for _ in range(self.num_envs)]
        if done or truncated:
            obs, _ = self.env.reset()
        self.last_obs = obs
        return self.last_obs, rewards, dones, infos

    def close(self): self.env.close()
    def get_attr(self, attr_name, indices=None): return [getattr(self.env, attr_name) for _ in range(self.num_envs)]
    def set_attr(self, attr_name, value, indices=None): setattr(self.env, attr_name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]
    def seed(self, seed=None): return [self.env.reset(seed=seed) for _ in range(self.num_envs)]
    def env_is_wrapped(self, wrapper_class, indices=None): return [False for _ in range(self.num_envs)]

class CurriculumCallback(BaseCallback):
    def __init__(self, n_agents: int, check_freq: int = 2048, patience_steps: int = 300000, verbose=1):
        super().__init__(verbose)
        self.n_agents = n_agents
        self.check_freq = check_freq
        self.history = []
        
    def _on_step(self) -> bool:
        relative_step = self.num_timesteps
        self.training_env.env_method('update_training_step', relative_step)
        
        if self.n_calls % self.check_freq == 0:
            info = self.locals['infos'][0]
            
            self.history.append({
                'step': self.num_timesteps,
                'success': info.get('success_rate', 0.0),
                'p2p_vol': info.get('p2p_volume', 0.0),
                'grid_redu': info.get('grid_reduction_percent', 0.0),
                'missed_pen': info.get('missed_trade_penalty', 0.0),
                'batt_pen': info.get('mean_battery_penalty', 0.0)
            })
            
            if self.verbose > 0:
                print(f"[v7-Emergence] N={self.n_agents} | Step: {self.num_timesteps} | P2P Vol: {info.get('p2p_volume', 0.0):.4f} | Success: {info.get('success_rate', 0.0):.1%} | Missed Pen: {info.get('missed_trade_penalty', 0):.3f}")
                
            if self.n_calls % (self.check_freq * 5) == 0:
                self._save_plots()

        return True

    def _save_plots(self):
        if not self.history: return
        df = pd.DataFrame(self.history)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(df['step'], df['p2p_vol']); axs[0, 0].set_title("P2P Volume (kW)")
        axs[0, 1].plot(df['step'], df['success']); axs[0, 1].set_title("Success Rate %")
        axs[1, 0].plot(df['step'], df['missed_pen']); axs[1, 0].set_title("Missed Trade Penalty")
        axs[1, 1].plot(df['step'], df['batt_pen']); axs[1, 1].set_title("Battery Penalty")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"training_n{self.n_agents}_emergence.png"))
        plt.close()

def run_v7_retraining():
    # Retraining Target: 1M steps for emergence
    training_steps = {4: 1000000}
    
    for n in AGENT_COUNTS:
        print(f"\n===== SLIM v7 EMERGENCE PHASE N={n} =====")
        train_env = VectorizedMultiAgentEnv(n_agents=n)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True) 
        
        model = PPO("MlpPolicy", train_env, verbose=1, 
                    learning_rate=3e-4, 
                    n_steps=2048, 
                    batch_size=256,
                    ent_coef=0.03) # High entropy for exploration noise discovery
            
        callback = CurriculumCallback(n_agents=n)
        model.learn(total_timesteps=training_steps[n], callback=callback)
        
        model.save(os.path.join(MODEL_DIR, f"ppo_n{n}_emergence"))
        train_env.save(os.path.join(MODEL_DIR, f"vec_normalize_n{n}_emergence.pkl"))
        train_env.close()

if __name__ == "__main__":
    run_v7_retraining()

if __name__ == "__main__":
    run_v7_retraining()
