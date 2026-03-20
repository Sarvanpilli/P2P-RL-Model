import os
import pandas as pd
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

def run_evaluation(env, model=None, steps=1000):
    obs = env.reset()
    
    metrics = {
        "p2p_volume": [],
        "profit": [],
        "grid_import": [],
        "grid_export": []
    }
    
    for t in range(steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline: Zero Action Policy (No battery logic, no P2P intent -> 100% grid interaction)
            action = np.zeros((1, env.envs[0].action_space.shape[0]))
            
        obs, rewards, dones, infos = env.step(action)
        info = infos[0] 
        
        metrics["p2p_volume"].append(info["p2p_volume"])
        metrics["profit"].append(info["profit"])
        metrics["grid_import"].append(info["grid_import"])
        metrics["grid_export"].append(info["grid_export"])
        
    return metrics

def evaluate(model_path, data_file="fixed_training_data.csv", steps=1000):
    def make_env():
         return EnergyMarketEnvSLIM(
            n_agents=4, 
            data_file=data_file,
            enable_safety=True,
            enable_p2p=True,
            seed=42
         )
         
    # 1. Evaluate RL Model
    env_rl = DummyVecEnv([make_env])
    
    # We load normalization bounds so evaluation inputs are perfectly matched to training scale
    try:
        norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        env_rl = VecNormalize.load(norm_path, env_rl)
        env_rl.training = False
        env_rl.norm_reward = False
        
        print(f"Loading trained RL model from {model_path}.zip ...")
        model = PPO.load(model_path, env=env_rl)
        print("Running RL Agent Evaluation against Dynamic Market...")
    except Exception as e:
        print(f"WARNING: Model or env normalization not found ({e}). Defaulting to random actions.")
        model = None
        
    rl_metrics = run_evaluation(env_rl, model=model, steps=steps)
    
    # 2. Evaluate Baseline (Rule-based / Zero Action grid default)
    env_base = DummyVecEnv([make_env])
    print("Running Baseline Evaluation (Zero Battery & Grid Default)...")
    base_metrics = run_evaluation(env_base, model=None, steps=steps)
    
    # 3. Compile Sequential Results
    df = pd.DataFrame({
        "timestep": np.arange(steps),
        "rl_p2p_volume": rl_metrics["p2p_volume"],
        "rl_profit": rl_metrics["profit"],
        "rl_grid_import": rl_metrics["grid_import"],
        "rl_grid_export": rl_metrics["grid_export"],
        "base_p2p_volume": base_metrics["p2p_volume"],
        "base_profit": base_metrics["profit"],
        "base_grid_import": base_metrics["grid_import"],
        "base_grid_export": base_metrics["grid_export"],
    })
    
    # Save Output CSV for Data Interrogation Check
    out_file = "research_q1/results/results_rl_vs_baseline.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved compiled episode dataset metrics to {out_file}")
    
    # Final Executive Summary
    print("\n--- FINAL EPISODE PERFORMANCE SUMMARY ---")
    print(f"Metric\t\t\tRL Agent (SLIM)\t\tBaseline (Grid)")
    print(f"Total P2P Volume\t{np.sum(rl_metrics['p2p_volume']):.2f} kWh\t\t{np.sum(base_metrics['p2p_volume']):.2f} kWh")
    print(f"Final Episode Profit\t${rl_metrics['profit'][-1]:.2f}\t\t\t${base_metrics['profit'][-1]:.2f}")
    print(f"Total Grid Import\t{np.sum(rl_metrics['grid_import']):.2f} kWh\t\t{np.sum(base_metrics['grid_import']):.2f} kWh")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="research_q1/models/rl_market_seed_42/final_model")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args.model_path, steps=args.steps)
