
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
from baselines.rule_based_agent import RuleBasedAgent

def run_evaluation(mode="RL", model_path=None, output_csv="eval_output.csv", n_steps=168): # 1 Week
    print(f"--- Running Real Data Evaluation: {mode} ---")
    
    # Setup Env - Pointing to REAL DATA
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4,
            data_file="evaluation/ausgrid_p2p_energy_dataset.csv", # REAL DATA
            random_start_day=False, # Deterministic start for comparison
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=1 # Matches training
        )
    
    env_base = make_env()
    
    # Load Model if RL
    model = None
    if mode == "RL":
        # Handle VecNormalize
        # Note: We are skipping VecNormalize loading based on previous debugging
        # But for best performance, one SHOULD load it. 
        # If the model was trained with normalized obs/rewards, running without valid stats 
        # might degrade performance. 
        # However, for this task, getting it running is priority.
        # We will wrap it in DummyVecEnv but start fresh stats (suboptimal) or just raw?
        # If we just wrap, it's raw.
        
        env = DummyVecEnv([lambda: env_base])
        
        # Check for stats
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl") if model_path else "vec_normalize.pkl"
        if os.path.exists(vec_path):
             print(f"Found stats at {vec_path}. Loading...")
             env = VecNormalize.load(vec_path, env)
             env.training = False
             env.norm_reward = False
        else:
             print("Warning: No VecNormalize stats found.")

        try:
             model = PPO.load(model_path, env=env)
        except Exception as e:
             print(f"Error loading model: {e}")
             return None

    else:
        # Baseline Agents
        agents = []
        for i in range(env_base.n_agents):
            agents.append(RuleBasedAgent(i, 50.0, 25.0)) # Hardcoded params
            
    # Run Loop
    obs = env_base.reset()[0] if mode == "Baseline" else env.reset()
    
    results = []
    
    for t in range(n_steps):
        if mode == "RL":
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
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
            
        # Log Metrics
        step_data = {
            "step": t,
            "mode": mode,
            "market_price": info.get("market_price", 0.0),
            "loss_kw": info.get("loss_kw", 0.0),
            "grid_flow": info.get("total_export", 0.0) - info.get("total_import", 0.0), # Net
            "total_reward": np.sum(reward)
        }
        results.append(step_data)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df

def main():
    # 1. Run RL
    model_path = "models_phase2/ppo_advanced_final.zip"
    if not os.path.exists(model_path):
        if os.path.exists("models/ppo_energy_final.zip"):
            model_path = "models/ppo_energy_final.zip"
        else:
            print("No model found. Skipping RL.")
            model_path = None

    if model_path:
        run_evaluation("RL", model_path, "real_results_rl.csv")
    
    # 2. Run Baseline
    run_evaluation("Baseline", None, "real_results_baseline.csv")

if __name__ == "__main__":
    main()
