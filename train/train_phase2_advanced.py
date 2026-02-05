# train_phase2_advanced.py
"""
Train a shared PPO policy (Stable-Baselines3) on the EnergyMarketEnvAdvanced.
Includes Regulatory Ramp Rates, Quadratic Losses, and Active Safety Guard.

Run:
    python train/train_phase2_advanced.py --timesteps 100000 --seed 42
"""
import argparse
import os
import sys
import numpy as np
import torch
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from train.energy_env_advanced import EnergyMarketEnvAdvanced
except ImportError:
    from energy_env_advanced import EnergyMarketEnvAdvanced

def make_env(rank, n_agents=4, n_prosumers=None, n_consumers=None, seed=0, forecast_horizon=1):
    def _init():
        # Initialize Advanced Env with strict constraints
        env = EnergyMarketEnvAdvanced(
            n_agents=n_agents,
            n_prosumers=n_prosumers,
            n_consumers=n_consumers,
            forecast_horizon=forecast_horizon,
            seed=seed + rank,
            data_file="test_day_profile.csv",
            # Phase 2 Configs
            ramp_limit_kw_per_hour=50.0,
            grid_resistance_ohms=0.05,
            voltage_penalty_coeff=1.0
        )
        env = Monitor(env)
        return env
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str, default="models_phase2")
    parser.add_argument("--log_dir", type=str, default="logs_phase2")
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Environment Setup
    # 4 Parallel Envs
    num_cpu = 4
    print(f"--- PHASE 2: TRAINING ADVANCED ENV ({num_cpu} parallel) ---")
    env = DummyVecEnv([make_env(i, n_agents=args.n_agents, seed=seed) for i in range(num_cpu)])

    # Normalize
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # Model Setup
    policy_kwargs = dict(net_arch=dict(pi=[400, 300], vf=[400, 300]))
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,   # Faster updates for debug
        batch_size=256, 
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        policy_kwargs=policy_kwargs
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=args.model_dir,
        name_prefix="ppo_advanced"
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # Save
    final_path = os.path.join(args.model_dir, "ppo_advanced_final")
    model.save(final_path)
    
    stats_path = os.path.join(args.model_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"Training completed. Saved to {final_path}")

if __name__ == "__main__":
    main()
