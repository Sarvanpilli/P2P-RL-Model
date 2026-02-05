# train_sb3_ppo.py
"""
Train a shared PPO policy (Stable-Baselines3) controlling all agents in the
energy_env_improved.Environment (flattened obs/actions).

Features:
- CLI arguments for seed, timesteps, overfit mode.
- Saves VecNormalize statistics.
- Deterministic training option.

Run:
    python train/train_sb3_ppo.py --timesteps 100000 --seed 42
"""
import argparse
import os
import sys
import numpy as np
import torch
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path to find 'train' module if running from train/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from train.energy_env_robust import EnergyMarketEnvRobust
except ImportError:
    from energy_env_robust import EnergyMarketEnvRobust

def make_env(rank, n_agents=4, n_prosumers=None, n_consumers=None, seed=0, forecast_horizon=1):
    def _init():
        # Initialize Robust Env with Data File
        env = EnergyMarketEnvRobust(
            n_agents=n_agents,
            n_prosumers=n_prosumers, # explicit
            n_consumers=n_consumers, # explicit
            forecast_horizon=forecast_horizon, 
            seed=seed + rank,
            data_file="test_day_profile.csv"
        )
        env = Monitor(env)
        return env
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining lies between 1.0 and 0.0.
        """
        return progress_remaining * initial_value
    return func

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=4, help="Total Agents (Deprecated, use P+C)")
    parser.add_argument("--n_prosumers", type=int, default=None, help="Number of Prosumers")
    parser.add_argument("--n_consumers", type=int, default=None, help="Number of Consumers")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overfit", action="store_true", help="Run in overfit mode (deterministic env)")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for tensorboard logs")
    args = parser.parse_args()

    # Set seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Environment Setup
    # Resolve Agent Counts
    if args.n_prosumers is not None and args.n_consumers is not None:
        total_agents = args.n_prosumers + args.n_consumers
        n_p = args.n_prosumers
        n_c = args.n_consumers
    else:
        total_agents = args.n_agents
        n_p = None
        n_c = None

    if args.overfit:
        print(f"--- OVERFIT MODE: Deterministic Environment (Seed {seed}) ---")
        env = make_env(0, n_agents=total_agents, n_prosumers=n_p, n_consumers=n_c, seed=seed)()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
    else:
        # Vectorized Environment
        num_cpu = 4
        print(f"--- NORMAL MODE: {num_cpu} Parallel Environments (Agents: {total_agents}) ---")
        env = DummyVecEnv([make_env(i, n_agents=total_agents, n_prosumers=n_p, n_consumers=n_c, seed=seed) for i in range(num_cpu)])

    # Normalize Observations and Rewards
    # Critical for PPO stability
    # We save this wrapper's stats later
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    # Model Setup - Optimized for Performance
    # Larger Net: [400, 300]
    # Higher Entropy: 0.01 (Exploration)
    # Larger Batch: 256
    policy_kwargs = dict(net_arch=dict(pi=[400, 300], vf=[400, 300]))
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=linear_schedule(3e-4), # Decay to 0
        n_steps=4096,   # Longer horizon
        batch_size=256, # Stable gradients
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Force exploration
        seed=seed,
        policy_kwargs=policy_kwargs
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=args.model_dir,
        name_prefix="ppo_energy"
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    # Save Final Model
    final_path = os.path.join(args.model_dir, "ppo_energy_final")
    model.save(final_path)
    
    # Save VecNormalize Stats
    stats_path = os.path.join(args.model_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"Training completed. Model saved to {final_path}")
    print(f"Normalization stats saved to {stats_path}")

if __name__ == "__main__":
    main()
