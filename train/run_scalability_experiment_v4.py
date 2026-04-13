
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
AGENT_COUNTS = [4, 8, 12, 16]
R_PROSUMER = 0.6
RESULTS_DIR = "research_q1/results/scalability_v4"
MODEL_DIR = "models_scalable_v4"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class VectorizedMultiAgentEnv(VecEnv):
    """
    Transparently presents a single multi-agent EnergyMarketEnv Robust 
    as a Vectorized environment for SB3.
    """
    def __init__(self, n_agents=4):
        n_pro = int(n_agents * R_PROSUMER)
        n_con = n_agents - n_pro
        self.env = EnergyMarketEnvRobust(n_prosumers=n_pro, n_consumers=n_con)
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
    """
    Step 11: Main Infrastructure
    1. Updates Environment global step for Decay Logic.
    2. Logs Liquidity and Market Metrics.
    3. Handles Early Stopping if plateaued.
    """
    def __init__(self, n_agents: int, check_freq: int = 2048, patience_steps: int = 50000, verbose=1):
        super().__init__(verbose)
        self.n_agents = n_agents
        self.check_freq = check_freq
        self.patience_steps = patience_steps
        self.best_success = -float('inf')
        self.best_grid_dep = float('inf')
        self.steps_without_improvement = 0
        self.history = []
        
    def _on_step(self) -> bool:
        # Sync RELATIVE Global Step to Environment for Epsilon/Margin/Noise Decay
        # This ensures curriculum resets for each N scale.
        relative_step = self.n_calls * self.training_env.num_envs
        self.training_env.env_method('update_training_step', relative_step)
        
        if self.n_calls % self.check_freq == 0:
            current_global_step = self.num_timesteps # For logging (absolute)
            info = self.locals['infos'][0]
            success = info.get('last_success_rate', 0.0)
            grid_dep = info.get('rolling_grid_dependency', 1.0)
            liquidity = info.get('liquidity', 0.0)
            eps = info.get('curriculum_epsilon', 0.15)
            
            # Log for plotting later
            self.history.append({
                'step': current_global_step,
                'success': success,
                'grid_dep': grid_dep,
                'liquidity': liquidity,
                'epsilon': eps
            })
            
            improved = False
            if success > self.best_success + 1e-4:
                self.best_success = success
                improved = True
            if grid_dep < self.best_grid_dep - 1e-4:
                self.best_grid_dep = grid_dep
                improved = True
                
            if improved: self.steps_without_improvement = 0
            else: self.steps_without_improvement += self.check_freq
                
            if self.verbose > 0:
                print(f"[Curriculum] Step: {current_global_step}, Success: {success:.3f}, Liq: {liquidity:.3f}, EPS: {eps:.3f}, Idle: {self.steps_without_improvement}")
                
            # Periodic Plotting (every 10 checks = ~20k steps)
            if self.n_calls % (self.check_freq * 10) == 0:
                self._save_plot()

            if self.steps_without_improvement >= self.patience_steps:
                print(f"Early stopping at {current_global_step} due to plateau.")
                return False
        return True

    def _save_plot(self):
        """Helper to generate plots dynamically."""
        if not self.history: return
        generate_step_plots(self.history, self.n_agents)

    def _on_training_end(self) -> None:
        pass # The main script calls generate_step_plots correctly

def generate_step_plots(history, n_agents):
    """Step 10: Validation Plots."""
    if not history: return
    df = pd.DataFrame(history)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Curriculum Market Relaxation - N={n_agents} Agents", fontsize=16)
    
    axs[0, 0].plot(df['step'], df['success'], color='blue', label='Trade Success')
    axs[0, 0].set_title("Trade Success Rate")
    axs[0, 0].set_ylabel("Rate")
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(df['step'], df['epsilon'], color='red', label='Epsilon Tolerance')
    axs[0, 1].set_title("Matching Epsilon (Curriculum)")
    axs[0, 1].set_ylabel("Price Tolerance ($/kWh)")
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(df['step'], df['liquidity'], color='green', label='Liquidity')
    axs[1, 0].set_title("Market Liquidity (Matched/Requested)")
    axs[1, 0].set_ylabel("Liquidity Ratio")
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(df['step'], df['grid_dep'], color='orange', label='Grid Dependency')
    axs[1, 1].set_title("Grid Dependency")
    axs[1, 1].set_ylabel("Ratio")
    axs[1, 1].set_xlabel("Steps")
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, f"training_curves_n{n_agents}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plots saved to {plot_path}")

def evaluate_scalability(model, n_agents, steps=2000):
    print(f"--- Final Evaluation N={n_agents} ---")
    eval_env = VectorizedMultiAgentEnv(n_agents=n_agents)
    # Set step to high value to ensure STRICT constraints (progress=1.0)
    eval_env.set_attr('total_training_steps', 1000000) 
    
    obs = eval_env.reset()
    p2p_history, dep_history, success_history = [], [], []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = eval_env.step(action)
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
    training_steps = {4: 100000, 8: 200000, 12: 300000, 16: 300000}
    
    for n in AGENT_COUNTS:
        print(f"\n===== CURRICULUM RECOVERY N={n} =====")
        train_env = VectorizedMultiAgentEnv(n_agents=n)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4)
        if current_model is not None:
            model.set_parameters(current_model.get_parameters())
            print(f"Policy weights transferred to N={n}.")
            
        callback = CurriculumCallback(n_agents=n, patience_steps=50000 if n<16 else 100000)
        model.learn(total_timesteps=training_steps[n], callback=callback)
        
        # Save results
        model.save(os.path.join(MODEL_DIR, f"ppo_n{n}_v4"))
        generate_step_plots(callback.history, n)
        
        res = evaluate_scalability(model, n)
        summary.append(res)
        current_model = model
        train_env.close()
        
    df = pd.DataFrame(summary)
    print("\n\n" + "="*50 + "\nCURRICULUM RECOVERY RESULTS\n" + "="*50)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "scalability_metrics_v4.csv"), index=False)
    return df

if __name__ == "__main__":
    run_scalability_sweep()
