
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

def train_slim():
    # Configuration
    N_AGENTS = 4
    TOTAL_TIMESTEPS = 300_000
    SEED = 42
    LOG_DIR = "research_q1/models/slim_ppo"
    TENSORBOARD_LOG = "research_q1/logs/slim_ppo_tensorboard"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    print(f"=== Starting SLIM PPO Training (N={N_AGENTS}) ===")
    
    # Environment Setup
    def make_env():
        env = EnergyMarketEnvSLIM(
            n_agents=N_AGENTS,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            seed=SEED
        )
        return env
        
    env = DummyVecEnv([make_env])
    env = VecMonitor(env) # Logs rewards/ep_len
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=LOG_DIR,
        name_prefix="slim_ppo"
    )
    
    # Model Setup (similar to IPPO baseline for fair comparison)
    # Using slightly larger net if needed, but keeping standard for now
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=TENSORBOARD_LOG,
        seed=SEED,
        device="auto"
    )
    


    # Training
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
            progress_bar=True
        )
        print("Training Finished.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:

        # Save Final Model and Stats
        model.save(f"{LOG_DIR}/slim_ppo_final")
        env.save(f"{LOG_DIR}/vec_normalize.pkl")
        print(f"Model saved to {LOG_DIR}")

if __name__ == "__main__":
    train_slim()
