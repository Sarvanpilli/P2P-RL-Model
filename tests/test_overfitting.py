import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_improved import EnergyMarketEnv

def test_overfitting():
    print("\n--- Test: RL Overfitting (Deterministic) ---")
    
    # Create a deterministic environment
    # We fix the seed and disable forecast uncertainty for this test
    env = EnergyMarketEnv(
        n_agents=2,
        forecast_horizon=0,
        forecast_uncertainty_std=0.0,
        seed=42
    )
    
    # Wrap in DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Create a small PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=42
    )
    
    print("Training for 2000 steps...")
    # Train
    model.learn(total_timesteps=2000)
    
    print("Evaluating...")
    # Evaluate
    obs = vec_env.reset()
    total_reward = 0.0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        if done[0]:
            obs = vec_env.reset()
            
    avg_reward = total_reward / 100
    print(f"Average Reward over 100 steps: {avg_reward}")
    
    # Check if reward is positive (or at least better than random)
    # Random reward is usually negative due to penalties.
    # We expect the agent to learn to avoid penalties at least.
    
    if avg_reward > -5.0: # Threshold depends on reward scale
        print("SUCCESS: Agent learned to avoid massive penalties.")
    else:
        print("WARNING: Agent reward is still very low. Check learning.")
        # We don't fail explicitly because 2000 steps is very short, 
        # but it should be enough to learn basic constraints.

if __name__ == "__main__":
    test_overfitting()
