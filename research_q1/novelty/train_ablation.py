
import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

def train_ablation(n_agents, total_timesteps, enable_safety, enable_p2p):
    # Configuration
    SEED = 42
    
    # Construct descriptive log directory
    suffix = []
    if not enable_safety: suffix.append("NoSafety")
    if not enable_p2p: suffix.append("NoP2P")
    
    config_name = f"N{n_agents}" + ("_" + "_".join(suffix) if suffix else "_FullSLIM")
    
    LOG_DIR = f"research_q1/models/slim_ablation_{config_name}"
    TENSORBOARD_LOG = f"research_q1/logs/slim_ablation_tensorboard"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    print(f"=== Starting Ablation Training: {config_name} ===")
    print(f"Safety: {enable_safety}, P2P: {enable_p2p}")
    
    # Environment Setup
    def make_env():
        env = EnergyMarketEnvSLIM(
            n_agents=n_agents,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            enable_safety=enable_safety, # Ablation flag
            enable_p2p=enable_p2p,       # Ablation flag
            seed=SEED
        )
        return env
        
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=LOG_DIR,
        name_prefix=f"ppo_{config_name}"
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
        tensorboard_log=TENSORBOARD_LOG,
        seed=SEED,
        device="auto"
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True,
            tb_log_name=f"PPO_{config_name}"
        )
        print(f"Training Finished for {config_name}.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        model.save(f"{LOG_DIR}/final_model")
        env.save(f"{LOG_DIR}/vec_normalize.pkl")
        print(f"Model saved to {LOG_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SLIM Agent for Ablation Studies")
    parser.add_argument("--n_agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps")
    parser.add_argument("--no_safety", action="store_true", help="Disable Safety Filter")
    parser.add_argument("--no_p2p", action="store_true", help="Disable Liquidity Pool (P2P)")
    
    args = parser.parse_args()
    
    # Invert flags because script args are negative (no_safety) but env args are positive (enable_safety)
    train_ablation(
        n_agents=args.n_agents,
        total_timesteps=args.timesteps,
        enable_safety=not args.no_safety,
        enable_p2p=not args.no_p2p
    )
