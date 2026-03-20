"""
train_scalability.py

Trains the SLIM MARL policy for varying numbers of agents (N=4, 6, 8, 10).
This script uses the robust EnergyMarketEnvRobust which now dynamically
cycles through the 4 core agent archetypes (Solar, Wind, EV, Standard).

Models are saved to:
    models_scalability/ppo_N4.zip
    models_scalability/ppo_N6.zip
    ...

Run this before evaluate_scalability.py if the models do not already exist.
"""

import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.energy_env_robust import EnergyMarketEnvRobust

def train_for_n_agents(n_agents: int, timesteps: int = 150_000, seed: int = 42):
    print(f"\n{'='*50}")
    print(f" TRAINING PPO FOR N={n_agents} AGENTS")
    print(f"{'='*50}")
    
    save_dir = "models_scalability"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"ppo_N{n_agents}")

    # Set up parallel environments
    n_envs = 4
    
    def make_env():
        # Using a fixed seed for reproducible training
        return EnergyMarketEnvRobust(n_agents=n_agents, random_start_day=True)
        
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        seed=seed,
        tensorboard_log=f"logs/scalability_tb/N{n_agents}/"
    )
    
    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO models for scalability testing")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per model")
    args = parser.parse_args()

    agent_counts = [4, 6, 8, 10]
    
    for n in agent_counts:
        train_for_n_agents(n_agents=n, timesteps=args.timesteps)
        
    print("\nAll models trained successfully. Run `python evaluate_scalability.py` next.")
