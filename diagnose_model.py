
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def diagnose():
    model_path = "models_phase4/ppo_predictive.zip"
    print(f"Diagnosing {model_path}...")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    print("1. Creating Env...")
    try:
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file="evaluation/ausgrid_p2p_energy_dataset.csv",
            enable_predictive_obs=True
        )
        print("   Env created.")
    except Exception as e:
        print(f"   Env creation failed: {e}")
        return

    print("2. Wrapping Env...")
    vec_env = DummyVecEnv([lambda: env])
    
    print("3. Attempting PPO Load...")
    try:
        model = PPO.load(model_path, env=vec_env)
        print(f"   PPO Load Success. Policy: {model.policy}")
    except Exception as e:
        print(f"   PPO Load Failed: {e}")
        
    print("4. Attempting RecurrentPPO Load (if module exists)...")
    try:
        from sb3_contrib import RecurrentPPO
        try:
            model_rnn = RecurrentPPO.load(model_path, env=vec_env)
            print(f"   RecurrentPPO Load Success. Policy: {model_rnn.policy}")
        except Exception as e:
            print(f"   RecurrentPPO Load Failed: {e}")
    except ImportError:
        print("   sb3_contrib not installed.")

    print("5. Testing Step...")
    try:
        obs = vec_env.reset()
        print(f"   Reset done. Obs shape: {obs.shape}")
        
        if 'model' in locals():
            action, _ = model.predict(obs)
            print("   PPO Predict done.")
            obs, reward, done, info = vec_env.step(action)
            print("   Step done.")
    except Exception as e:
        print(f"   Step Failed: {e}")

if __name__ == "__main__":
    diagnose()
