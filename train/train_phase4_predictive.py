
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
    print("WARNING: sb3-contrib not found. Falling back to simple PPO with MlpPolicy.")

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust

def make_env():
    # Phase 4 Configuration
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="merged_dataset_phase2.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=1,
        enable_predictive_obs=True # Enable Phase 4 features
    )
    return env

def train():
    log_dir = "./tensorboard_logs/"
    model_dir = "models_phase4"
    os.makedirs(model_dir, exist_ok=True)
    
    # from stable_baselines3.common.vec_env import DummyVecEnv as VecEnv # Debugging
    from stable_baselines3.common.vec_env import SubprocVecEnv as VecEnv
    
    # 1. Parallelize (Vectorization)
    n_envs = 4 
    env = VecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    # Note: norm_obs=False because we manually normalized in _get_obs to [-1, 1]
    
    # 2. Model Setup
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        # LSTM specific would go here if kwargs supported it similarly
    )
    
    if USING_RECURRENT:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=1e-4, # Lower LR for Recurrent
            n_steps=512, # per env -> 2048 total buffer
            batch_size=128,
            n_epochs=10,
            gamma=0.999, # Long horizon
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
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=log_dir
        )
    
    # 3. Train
    TIMESTEPS = 200000 
    print(f"Starting Phase 4 Training for {TIMESTEPS} steps...")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="ppo_predictive")
    
    # 4. Save
    model_path = os.path.join(model_dir, "ppo_predictive")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")
    env.save(vec_path)
    print(f"Stats saved to {vec_path}")

if __name__ == "__main__":
    train()
