"""
Phase 5 Recovery Training Script (From Scratch)

Since the 150k checkpoint has incompatible action space (3D vs 2D),
this script trains a new model from scratch using the recovery environment.

Strategy:
- Start fresh with recovery environment
- Use stability hyperparameters
- Train for 200k steps to reach 150k baseline performance
- Monitor P2P trading activity
"""

import os
import sys
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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


def make_env(rank: int, seed: int = 0):
    """Create a single environment instance"""
    def _init():
        env = EnergyMarketEnvRecovery(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            forecast_noise_std=0.05,
            diversity_mode=True,
            seed=seed + rank
        )
        env = Monitor(env)
        return env
    return _init


def main():
    """Train Phase 5 Recovery model from scratch"""
    
    print("\n" + "="*60)
    print("PHASE 5 RECOVERY TRAINING (FROM SCRATCH)")
    print("="*60)
    print("Strategy: Liquidity-First P2P + Positive Rewards")
    print("Starting: New model (150k checkpoint incompatible)")
    print("="*60 + "\n")
    
    # === CONFIGURATION ===
    OUTPUT_DIR = "models_phase5_recovery"
    LOG_DIR = "./tensorboard_logs/"
    
    N_ENVS = 4
    TOTAL_TIMESTEPS = 200000  # Train to 200k to surpass 150k baseline
    CHECKPOINT_FREQ = 25000
    
    # === STABILITY HYPERPARAMETERS ===
    LEARNING_RATE = 5e-5
    ENT_COEF = 0.05
    N_STEPS = 4096
    BATCH_SIZE = 512
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    
    print("Hyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Entropy Coef: {ENT_COEF}")
    print(f"  N Steps: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Parallel Envs: {N_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print()
    
    # === CREATE DIRECTORIES ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # === CREATE PARALLEL ENVIRONMENTS ===
    print("Creating parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # === CREATE VecNormalize ===
    print("Creating VecNormalize wrapper...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # === CREATE NEW MODEL ===
    print("\nCreating new model from scratch...")
    
    if USING_RECURRENT:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            vf_coef=0.5,
            max_grad_norm=0.5,
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
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device='auto'
        )
    
    print("✓ Model created successfully!")
    
    # === SETUP CALLBACKS ===
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,
        save_path=OUTPUT_DIR,
        name_prefix="ppo_recovery",
        verbose=1
    )
    
    # === TRAINING ===
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoint frequency: {CHECKPOINT_FREQ:,} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nMonitor these metrics in TensorBoard:")
    print("  - reward/p2p_bonus_mean (should increase)")
    print("  - p2p_volume (should be > 0)")
    print("  - mean_soc_pct (should stabilize > 0.3)")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="ppo_recovery_scratch",
            callback=checkpoint_callback,
            reset_num_timesteps=True
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        # === SAVE FINAL MODEL ===
        final_steps = model.num_timesteps
        final_model_name = f"ppo_recovery_{final_steps}_steps"
        model.save(os.path.join(OUTPUT_DIR, final_model_name))
        env.save(os.path.join(OUTPUT_DIR, "vec_normalize.pkl"))
        
        print(f"✓ Final model saved: {final_model_name}.zip")
        print(f"✓ Total timesteps: {final_steps:,}")
        print(f"✓ VecNormalize stats saved")
        print("="*60 + "\n")
        
        print("Next Steps:")
        print("1. Evaluate the recovery model")
        print("2. Check P2P trading activity (should be > 0)")
        print("3. Compare SoC stability with original Phase 5")
        print("4. If successful, continue training to 500k")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving current model...")
        
        current_steps = model.num_timesteps
        interrupt_model_name = f"ppo_recovery_{current_steps}_steps_interrupted"
        model.save(os.path.join(OUTPUT_DIR, interrupt_model_name))
        env.save(os.path.join(OUTPUT_DIR, "vec_normalize.pkl"))
        
        print(f"✓ Model saved: {interrupt_model_name}.zip")
        print(f"✓ Timesteps completed: {current_steps:,}")
        
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nAttempting to save model...")
        try:
            error_model_name = f"ppo_recovery_{model.num_timesteps}_steps_error"
            model.save(os.path.join(OUTPUT_DIR, error_model_name))
            env.save(os.path.join(OUTPUT_DIR, "vec_normalize.pkl"))
            print(f"✓ Model saved: {error_model_name}.zip")
        except:
            print("❌ Could not save model")


if __name__ == "__main__":
    main()
