"""
Phase 5 Recovery Training (Single Environment - No Multiprocessing)

Simplified training script without parallel environments to avoid multiprocessing issues.
"""

import os
import sys
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
    print("✓ Using RecurrentPPO (LSTM) Policy")
except ImportError:
    USING_RECURRENT = False
    from stable_baselines3 import PPO
    print("⚠ RecurrentPPO not available, using standard PPO")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_recovery import EnergyMarketEnvRecovery


def make_env():
    """Create environment"""
    env = EnergyMarketEnvRecovery(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4,
        forecast_noise_std=0.05,
        diversity_mode=True,
        seed=42
    )
    env = Monitor(env)
    return env


def main():
    print("\n" + "="*60)
    print("PHASE 5 RECOVERY TRAINING (SINGLE ENV)")
    print("="*60)
    print("Strategy: Liquidity-First P2P + Positive Rewards")
    print("="*60 + "\n")
    
    # Configuration
    OUTPUT_DIR = "models_phase5_recovery"
    LOG_DIR = "./tensorboard_logs/"
    
    TOTAL_TIMESTEPS = 50000  # Shorter test run
    CHECKPOINT_FREQ = 10000
    
    # Hyperparameters
    LEARNING_RATE = 5e-5
    ENT_COEF = 0.05
    N_STEPS = 2048  # Smaller for single env
    BATCH_SIZE = 256
    
    print("Hyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Entropy Coef: {ENT_COEF}")
    print(f"  N Steps: {N_STEPS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create single environment with DummyVecEnv
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    
    print("Creating VecNormalize wrapper...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    print("\nCreating new model...")
    if USING_RECURRENT:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENT_COEF,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device='auto'
        )
    else:
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
    
    print("✓ Model created!\n")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=OUTPUT_DIR,
        name_prefix="ppo_recovery_single",
        verbose=1
    )
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="ppo_recovery_single",
            callback=checkpoint_callback
        )
        
        print("\n✅ TRAINING COMPLETE!")
        
        final_model_name = f"ppo_recovery_single_{model.num_timesteps}_steps"
        model.save(os.path.join(OUTPUT_DIR, final_model_name))
        env.save(os.path.join(OUTPUT_DIR, "vec_normalize_single.pkl"))
        
        print(f"✓ Model saved: {final_model_name}.zip")
        print(f"✓ Total timesteps: {model.num_timesteps:,}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted")
        model.save(os.path.join(OUTPUT_DIR, f"ppo_recovery_single_{model.num_timesteps}_interrupted"))
        env.save(os.path.join(OUTPUT_DIR, "vec_normalize_single.pkl"))
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
