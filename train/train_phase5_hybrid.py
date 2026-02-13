
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
    # Phase 5 Hybrid Configuration
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4, # 4-hr lookahead
        
        # Hybrid Settings (implied by file loading logic in Env)
        enable_predictive_obs=True, 
        forecast_noise_std=0.05, 
        diversity_mode=True # Activates Hybrid Archetypes
    )
    return env

def train():
    log_dir = "./tensorboard_logs/"
    model_dir = "models_phase5_hybrid"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Parallelize (8 Envs as requested)
    # Windows requires 'spawn' usually, handled by SB3/Multiprocessing
    # Issues with EOFError -> Switch to DummyVecEnv for stability check
    n_envs = 1 # Debug Mode
    # env = SubprocVecEnv([make_env for _ in range(n_envs)])
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # 2. Model Setup
    # "LSTM hidden states to 256"
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=256,
        n_lstm_layers=1 # Standard
    )
    
    lr = 1e-4 # Standard
    
    if USING_RECURRENT:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            n_steps=512, # 8*512 = 4096 buffer
            batch_size=256,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device='auto'
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # 3. Train
    TIMESTEPS = 500000 
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs, # Save roughly every 50k steps
        save_path=model_dir,
        name_prefix="ppo_hybrid"
    )
    
    print(f"Starting Phase 5 Hybrid Training for {TIMESTEPS} steps...")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="ppo_hybrid", callback=checkpoint_callback)
    
    # 4. Save
    model.save(os.path.join(model_dir, "ppo_hybrid_final"))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print("Training Complete.")

if __name__ == "__main__":
    train()
