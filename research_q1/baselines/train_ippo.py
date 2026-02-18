
import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path to import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train.energy_env_recovery import EnergyMarketEnvRecovery

def make_env(seed=42):
    """Create environment wrapped for SB3"""
    env = EnergyMarketEnvRecovery(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4,
        forecast_noise_std=0.05,
        diversity_mode=True,
        seed=seed
    )
    env = Monitor(env)
    return env

def train_ippo_baseline():
    print("\n" + "="*60)
    print("RESEARCH BASELINE: INDEPENDENT PPO (IPPO)")
    print("="*60)
    print("Goal: Establish standard RL performance benchmark.")
    print("="*60 + "\n")
    
    # Configuration
    OUTPUT_DIR = "research_q1/models/ippo_baseline"
    LOG_DIR = "./tensorboard_logs/ippo_baseline"
    
    TOTAL_TIMESTEPS = 100000 # Enough to see convergence
    CHECKPOINT_FREQ = 10000
    
    # Hyperparameters (Standard PPO Defaults)
    # We intentionally do NOT purely tune these for the baseline to represent "Standard RL"
    LEARNING_RATE = 3e-4 
    N_STEPS = 2048
    BATCH_SIZE = 64
    ENT_COEF = 0.0 # Standard is usually 0.0 or low
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create Env
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device='auto'
    )
    
    print("Starting Standard PPO Training...")
    
    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=OUTPUT_DIR,
        name_prefix="ippo_baseline"
    )
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="ippo_run"
        )
        
        print("\n✅ IPPO BASELINE TRAINING COMPLETE!")
        model.save(os.path.join(OUTPUT_DIR, "ippo_final"))
        env.save(os.path.join(OUTPUT_DIR, "vec_normalize.pkl"))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_ippo_baseline()
