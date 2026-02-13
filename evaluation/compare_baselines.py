
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

def run_evaluation(mode="RL", model_path=None, output_csv="eval_output.csv", n_steps=24):
    print(f"--- Running Evaluation: {mode} ---")
    
    # Setup Env
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4,
            data_file="test_day_profile.csv", # Or merged
            random_start_day=False,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=1
        )
    
    env_base = make_env()
    
    # Load Model if RL
    model = None
    if mode == "RL":
        # Handle VecNormalize
        # Assuming we have statistics or just run raw for comparison if no stats file passed
        # Ideally load stats.
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl") if model_path else "vec_normalize.pkl"
        
        env = DummyVecEnv([lambda: env_base])
        if os.path.exists(vec_path):
            print(f"Found stats at {vec_path}, but SKIPPING load to debug dimensions.")
            # print("Loading VecNormalize stats...")
            # env = VecNormalize.load(vec_path, env)
            # env.training = False
            # env.norm_reward = False
        else:
            print("Warning: No VecNormalize stats found. RL performance might vary.")
            
        print(f"--- Space Debug ---")
        print(f"Env Obs Shape: {env.observation_space.shape}")
        
        try:
             model = PPO.load(model_path, env=env)
        except ValueError as e:
             print(f"FAILED TO LOAD MODEL due to space mismatch.")
             # Try loading without env to inspect
             model = PPO.load(model_path)
             print(f"Model Expected Obs Shape: {model.observation_space.shape}")
             raise e
    else:
        # Baseline Agents
        agents = []
        for i in range(env_base.n_agents):
            agents.append(RuleBasedAgent(i, 50.0, 25.0)) # Hardcoded params for now
            
    # Run Loop
    obs = env_base.reset()[0] if mode == "Baseline" else env.reset()
    
    results = []
    
    for t in range(n_steps):
        if mode == "RL":
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            # Unnormalize reward?
            # We care about raw metrics in info
        else:
            # Baseline Loop
            # Construct composite action
            actions = []
            # Obs is flat. Need to slice per agent.
            # Base dim 8 + forecasts.
            # EnergyMarketEnvRobust has obs_dim property? Yes.
            # But let's cheat and use env.nodes internal state for cleaner code?
            # No, let's try to inspect obs size.
            obs_per_agent = len(obs) // env_base.n_agents
            
            for i in range(env_base.n_agents):
                agent_obs = obs[i*obs_per_agent : (i+1)*obs_per_agent]
                act = agents[i].get_action(agent_obs, t)
                actions.append(act)
            
            flat_action = np.array(actions).flatten()
            obs, reward, done, trunc, info = env_base.step(flat_action)
            
        # Log Metrics
        # Info contains: market_price, loss_kw, total_export, total_import, trades_kw, battery_throughput...
        step_data = {
            "step": t,
            "mode": mode,
            "market_price": info.get("market_price", 0.0),
            "loss_kw": info.get("loss_kw", 0.0),
            "grid_flow": info.get("total_export", 0.0) - info.get("total_import", 0.0), # Net
            "total_reward": np.sum(reward) # Both paths now return array or scalar, handle carefully
        }
        results.append(step_data)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df

def main():
    # 1. Run RL
    # Path to model - update as needed
    model_path = "models_phase2/ppo_advanced_final.zip" # Default check
    if not os.path.exists(model_path):
        # Fallback to look for *any* model
        if os.path.exists("models/ppo_energy_final.zip"):
            model_path = "models/ppo_energy_final.zip"
        else:
            print("No model found. Skipping RL run.")
            model_path = None

    if model_path:
        df_rl = run_evaluation("RL", model_path, "results_rl.csv", n_steps=48)
    
    # 2. Run Baseline
    df_base = run_evaluation("Baseline", None, "results_baseline.csv", n_steps=48)
    
    # 3. Compare / Plot
    # Just simple print for now
    if model_path:
        print("\n--- Comparison (Avg over 48 steps) ---")
        print(f"RL Reward: {df_rl['total_reward'].mean():.2f} | Baseline Reward: {df_base['total_reward'].mean():.2f}")
        print(f"RL Loss: {df_rl['loss_kw'].mean():.4f} | Baseline Loss: {df_base['loss_kw'].mean():.4f}")
        print(f"RL Price Std: {df_rl['market_price'].std():.4f} | Baseline Price Std: {df_base['market_price'].std():.4f}")

if __name__ == "__main__":
    main()
