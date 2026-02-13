
import gym
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust

def make_env():
    # Phase 3 Configuration:
    # - Robust Environment with new Reward Logic (Grid Penalty)
    # - Real Data or Test Profile? 
    # Let's use the merged dataset if available, or test profile.
    data_file = "merged_dataset_phase2.csv"
    if not os.path.exists(data_file):
        data_file = "test_day_profile.csv"
        print("Warning: Merged dataset not found, using test profile.")
    
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file=data_file,
        random_start_day=True, # Randomize start day for robustness
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=1
    )
    return env

def train():
    log_dir = "./tensorboard_logs/"
    model_dir = "models_phase3"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Setup Env with Normalization
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Create Model
    # Using MlpPolicy with default params
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir
    )
    
    # 3. Train
    # Shorten for demonstration
    TIMESTEPS = 10000 
    print(f"Starting training for {TIMESTEPS} steps...")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="ppo_grid_aware")
    
    # 4. Save
    model_path = os.path.join(model_dir, "ppo_grid_aware")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save Normalization Stats
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")
    env.save(vec_path)
    print(f"Stats saved to {vec_path}")

if __name__ == "__main__":
    train()
