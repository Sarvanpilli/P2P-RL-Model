
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
from baselines.rule_based_agent import RuleBasedAgent

def run_evaluation(mode="RL", model_path=None, output_csv="eval_output.csv", n_steps=168): # 1 Week
    print(f"--- Running Eval: {mode} ---")
    
    # Setup Env - Pointing to REAL DATA
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4,
            data_file="evaluation/ausgrid_p2p_energy_dataset.csv",
            random_start_day=False, 
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=1 
        )
    
    env_base = make_env()
    
    # Load Model
    model = None
    env = None
    
    if "RL" in mode:
        env = DummyVecEnv([lambda: env_base])
        # Try loading stats
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl") if model_path else "vec_normalize.pkl"
        if os.path.exists(vec_path):
             print(f"Loading stats from {vec_path}...")
             env = VecNormalize.load(vec_path, env)
             env.training = False
             env.norm_reward = False
        else:
             print("Warning: No stats found.")
             
        try:
             model = PPO.load(model_path, env=env)
        except Exception as e:
             print(f"Error loading model {model_path}: {e}")
             return None

    else:
        # Baseline Agents
        agents = []
        for i in range(env_base.n_agents):
            agents.append(RuleBasedAgent(i, 50.0, 25.0)) 

    # Run Loop
    obs = env_base.reset()[0] if "Baseline" in mode else env.reset()
    
    results = []
    
    for t in range(n_steps):
        if "RL" in mode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            # reward is array, sum it
            total_r = np.sum(reward)
        else:
            # Baseline Loop
            actions = []
            obs_per_agent = len(obs) // env_base.n_agents
            for i in range(env_base.n_agents):
                agent_obs = obs[i*obs_per_agent : (i+1)*obs_per_agent]
                act = agents[i].get_action(agent_obs, t)
                actions.append(act)
            
            flat_action = np.array(actions).flatten()
            obs, reward, done, trunc, info = env_base.step(flat_action)
            total_r = np.sum(reward)
            
        # Metrics
        # Info has: market_price, loss_kw, total_export, total_import...
        step_data = {
            "step": t,
            "mode": mode,
            "market_price": info.get("market_price", 0.0),
            "loss_kw": info.get("loss_kw", 0.0),
            "net_grid_flow": info.get("total_export", 0.0) - info.get("total_import", 0.0),
            "total_import": info.get("total_import", 0.0),
            "total_reward": total_r
        }
        results.append(step_data)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df

def main():
    # 1. Phase 2 Model (Legacy)
    if os.path.exists("models_phase2/ppo_advanced_final.zip"):
        run_evaluation("RL_Phase2", "models_phase2/ppo_advanced_final.zip", "results_phase2.csv")
    
    # 2. Phase 3 Model (Grid Aware)
    if os.path.exists("models_phase3/ppo_grid_aware.zip"):
        run_evaluation("RL_Phase3", "models_phase3/ppo_grid_aware.zip", "results_phase3.csv")
    else:
        print("Phase 3 model not found yet.")

    # 3. Baseline
    run_evaluation("Baseline", None, "results_baseline.csv")

if __name__ == "__main__":
    main()
