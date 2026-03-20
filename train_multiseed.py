
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def make_env(seed=0):
    def _init():
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            enable_predictive_obs=True,
            diversity_mode=True
        )
        # env.seed(seed) # Gymnasium seed handled in reset
        return env
    return _init

def train_multiseed(seeds=[0, 1, 2, 3, 4], total_timesteps=500000):
    """Runs 5-seed training for SLIM model."""
    print(f"Starting 5-seed training (Total timesteps per seed: {total_timesteps})...")
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"TRAINING SEED: {seed}")
        print(f"{'='*60}")
        
        model_dir = f"models_slim/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 4 parallel envs per seed
        env = SubprocVecEnv([make_env(seed + i) for i in range(4)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            seed=seed,
            tensorboard_log=f"./tboard_slim/seed_{seed}/"
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=25000 // 4, # adjust for parallel envs
            save_path=model_dir,
            name_prefix="ppo_slim"
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback
        )
        
        # Save final model and normalization stats
        model.save(f"{model_dir}/best_model.zip")
        env.save(f"{model_dir}/vec_normalize.pkl")
        print(f"Seed {seed} complete. Model saved to {model_dir}/best_model.zip")

if __name__ == "__main__":
    # Run full 500k training per seed as per Prompt 3
    train_multiseed(total_timesteps=500000)
