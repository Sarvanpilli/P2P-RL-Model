    
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

def test_slim_env():
    print("\n Testing SLIM Environment...")
    
    # 1. Setup Env
    def make_env():
        return EnergyMarketEnvSLIM(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=True, 
            enable_ramp_rates=True,
            enable_losses=True, # Underlying physics options
             seed=42
        )
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available (best effort for obs)
    vec_norm_path = "research_q1/models/ippo_baseline/vec_normalize.pkl"
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded VecNormalize stats")
    
    # Load IPPO Model
    model_path = "research_q1/models/ippo_baseline/ippo_final"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)
        print("✓ Loaded IPPO Model")
    else:
        print("⚠ IPPO Model not found, using random actions")
        model = None
        
    # Run loop
    obs = env.reset()
    total_reward = 0.0
    
    print("-" * 40)
    print(f"{'Step':<5} | {'Reward':<10} | {'Info'}")
    print("-" * 40)
    
    for t in range(24):
        if model:
            action, _ = model.predict(obs, deterministic=True)
            # Fix batching for single env if needed
            if len(action.shape) == 1:
                action = [action]
        else:
            action = [env.action_space.sample()]
            
        obs, rewards, done, info = env.step(action)
        
        r = rewards[0]
        inf = info[0]
        
        total_reward += r
        print(f"{t:<5} | {r:<10.2f} | P2P:{inf.get('p2p_volume',0):.2f} Price:{inf.get('clearing_price',0):.2f}")
        
    print("-" * 40)
    print(f"Total Reward (24 steps): {total_reward:.2f}")
    
    # Access inner env stats
    inner_env = env.envs[0] # wrapper -> wrapper -> env
    # If Norm wrapper, env.venv.envs[0]?
    # VecNormalize wraps a VecEnv.
    # inner is env.venv.envs[0]
    
    # Access safety violations
    try:
        if hasattr(inner_env, 'safety_violations'):
            print(f"Safety Violations: {inner_env.safety_violations}")
        elif hasattr(env, 'get_attr'):
             # SB3 generic way
             viols = env.get_attr('safety_violations')[0]
             print(f"Safety Violations: {viols}")
    except:
        pass

if __name__ == "__main__":
    test_slim_env()
