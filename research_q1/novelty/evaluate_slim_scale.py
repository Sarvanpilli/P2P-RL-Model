
import os
import sys
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

def evaluate_slim(n_agents, model_path, episodes=10, enable_safety=True, enable_p2p=True):
    print(f"=== Evaluating SLIM Agent (N={n_agents}) ===")
    print(f"Model: {model_path}")
    print(f"Safety: {enable_safety}, P2P: {enable_p2p}")
    
    # recreate env
    def make_env():
        return EnergyMarketEnvSLIM(
            n_agents=n_agents,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            enable_safety=enable_safety,
            enable_p2p=enable_p2p,
            seed=42 # Fixed seed for eval
        )
        
    env = DummyVecEnv([make_env])
    # Load normalization stats if available
    norm_path = os.path.dirname(model_path) + "/vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"Loading normalization stats from {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = False # Don't update stats during eval
        env.norm_reward = False # valid for evaluation? usually yes, but we want raw reward for reporting?
        # actually if we want to see the Agent's perception, we need norm. 
        # But for reporting "Diff in $", we want unnormalized. 
        # VecNormalize un-normalizes return if we ask? 
        # standard PPO eval usually inspects the info dict or unnormalized env reward.
        # Let's keep norm_reward=False to get raw rewards in step returns?
        # NO, if the model expects normalized rewards (for Value func), it matters less for execution.
        # But for 'obs' it is critical.
    else:
        print("Warning: No normalization stats found!")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO.load(model_path)
    
    all_rewards = []
    total_violations = 0
    total_p2p_volume = 0.0
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Info is a list for VecEnv
            info_dict = info[0]
            
            # We want RAW profit (unnormalized)
            # slim_env returns scalar 'profit' in info dict which is sum of raw rewards
            raw_profit = info_dict.get('profit', reward[0]) # Fallback
            
            ep_reward += raw_profit
            
            # Tracking
            # We need to access the underlying env for safety violations
            # DummyVecEnv -> envs[0]
            base_env = env.envs[0]
            # But safety_violations is cumulative in base_env. 
            # We can diff it or just read at end.
            
            # Update P2P volume
            total_p2p_volume += info_dict.get('p2p_volume', 0.0)
            
        all_rewards.append(ep_reward)
        viz_env = env.envs[0]
        total_violations += viz_env.safety_violations
        
        print(f"Episode {ep+1}: Profit = ${ep_reward:.2f}")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_violations = total_violations / episodes
    avg_p2p = total_p2p_volume / episodes
    
    print("\n=== Results ===")
    print(f"Mean Daily Profit: ${mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Avg Safety Violations: {avg_violations:.2f}")
    print(f"Avg P2P Volume: {avg_p2p:.2f} kWh")
    
    return mean_reward, avg_violations, avg_p2p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no_safety", action="store_true")
    parser.add_argument("--no_p2p", action="store_true")
    
    args = parser.parse_args()
    
    evaluate_slim(
        n_agents=args.n_agents,
        model_path=args.model,
        episodes=args.episodes,
        enable_safety=not args.no_safety,
        enable_p2p=not args.no_p2p
    )
