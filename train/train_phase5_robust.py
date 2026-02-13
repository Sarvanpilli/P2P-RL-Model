
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
    print("Using RecurrentPPO (LSTM) Policy")
except ImportError:
    USING_RECURRENT = False
    print("WARNING: sb3-contrib not found. Falling back to PPO.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def make_env():
    # Phase 5 Configuration
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="merged_dataset_phase2.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=1,
        
        # Phase 5 Features
        enable_predictive_obs=True, # Base
        forecast_noise_std=0.05, # Stochasticity
        diversity_mode=True # Heterogeneity
    )
    return env

def train():
    log_dir = "./tensorboard_logs/"
    model_dir = "models_phase5"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Parallelize
    n_envs = 4
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    # from stable_baselines3.common.vec_env import DummyVecEnv
    # env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    
    # 2. Model Setup
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # Lower LR for noisy environment
    lr = 5e-5 
    
    if USING_RECURRENT:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            n_steps=512, 
            batch_size=128,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=log_dir
        )
    
    # 3. Train
    TIMESTEPS = 300000 # Longer training for robustness
    
    # Save a checkpoint every 20480 steps (approx 10 updates * 4 envs * 512 steps if using RecurrentPPO)
    checkpoint_callback = CheckpointCallback(
        save_freq=20480,
        save_path=model_dir,
        name_prefix="ppo_robust"
    )
    
    print(f"Starting Phase 5 Robust Training for {TIMESTEPS} steps...")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="ppo_robust", callback=checkpoint_callback)
    
    # 4. Save Final
    model_path = os.path.join(model_dir, "ppo_robust")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")
    env.save(vec_path)

if __name__ == "__main__":
    train()
