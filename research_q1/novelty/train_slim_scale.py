
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

def train_slim_scale(n_agents, total_timesteps):
    # Configuration
    SEED = 42
    LOG_DIR = f"research_q1/models/slim_ppo_N{n_agents}"
    TENSORBOARD_LOG = f"research_q1/logs/slim_ppo_scale_tensorboard"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    print(f"=== Starting SLIM PPO Training (N={n_agents}) ===")
    
    # Environment Setup
    def make_env():
        env = EnergyMarketEnvSLIM(
            n_agents=n_agents,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            seed=SEED
        )
        return env
        
    env = DummyVecEnv([make_env])
    env = VecMonitor(env) # Logs rewards/ep_len
    # Norm obs is crucial for scalability as state space grows
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=LOG_DIR,
        name_prefix=f"slim_ppo_N{n_agents}"
    )
    
    # Model Setup
    # For larger N, maybe increase net size? Keeping standard for now for fair comparison.
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
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True,
            tb_log_name=f"PPO_N{n_agents}"
        )
        print(f"Training Finished for N={n_agents}.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save Final Model and Stats
        model.save(f"{LOG_DIR}/slim_ppo_final")
        env.save(f"{LOG_DIR}/vec_normalize.pkl")
        print(f"Model saved to {LOG_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SLIM Agent with Scalable N")
    parser.add_argument("--n_agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps")
    
    args = parser.parse_args()
    
    train_slim_scale(args.n_agents, args.timesteps)
