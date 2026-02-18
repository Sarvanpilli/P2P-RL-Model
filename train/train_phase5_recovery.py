"""
Phase 5 Recovery Training Script

Resumes training from the 150k checkpoint (Gold Baseline) with:
1. Refactored environment (liquidity-first P2P)
2. Positive reward shaping
3. Stability-focused hyperparameters
4. Parallel environments for stable gradients

Goal: Fix policy collapse and enable P2P trading activity.
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
    """
    Create a single environment instance.
    
    Args:
        rank: Environment ID for parallel training
        seed: Random seed offset
    """
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
        env = Monitor(env)  # Wrap with Monitor for logging
        return env
    return _init


def main():
    """
    Phase 5 Recovery Training
    
    Strategy:
    - Load 150k checkpoint (best performing)
    - Use very low learning rate for stability
    - Increase entropy for exploration
    - Use 4 parallel environments for stable gradients
    - Train for 100k steps to test recovery
    """
    
    print("\n" + "="*60)
    print("PHASE 5 RECOVERY TRAINING")
    print("="*60)
    print("Strategy: Liquidity-First P2P + Positive Rewards")
    print("Checkpoint: ppo_hybrid_150000_steps.zip (Gold Baseline)")
    print("="*60 + "\n")
    
    # === CONFIGURATION ===
    CHECKPOINT_PATH = "models_phase5_hybrid/ppo_hybrid_150000_steps.zip"
    VEC_NORMALIZE_PATH = "models_phase5_hybrid/vec_normalize.pkl"
    OUTPUT_DIR = "models_phase5_recovery"
    LOG_DIR = "./tensorboard_logs/"
    
    N_ENVS = 4  # Parallel environments
    TOTAL_TIMESTEPS = 100000  # Test run (100k steps)
    CHECKPOINT_FREQ = 25000  # Save every 25k steps
    
    # === STABILITY HYPERPARAMETERS ===
    LEARNING_RATE = 5e-5      # Very low (was 3e-4)
    ENT_COEF = 0.05           # High exploration (was ~0.01)
    N_STEPS = 4096            # 2 days of data per update
    BATCH_SIZE = 512          # Larger batches for stability
    GAMMA = 0.99              # Standard discount
    GAE_LAMBDA = 0.95         # Standard GAE
    
    print("Hyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Entropy Coef: {ENT_COEF}")
    print(f"  N Steps: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Parallel Envs: {N_ENVS}")
    print()
    
    # === CREATE DIRECTORIES ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # === CREATE PARALLEL ENVIRONMENTS ===
    print("Creating parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # === LOAD OR CREATE VecNormalize ===
    if os.path.exists(VEC_NORMALIZE_PATH):
        print(f"Loading VecNormalize stats from: {VEC_NORMALIZE_PATH}")
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = True
        env.norm_reward = True
    else:
        print("Creating new VecNormalize wrapper")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # === LOAD CHECKPOINT ===
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
        
        if USING_RECURRENT:
            model = RecurrentPPO.load(
                CHECKPOINT_PATH,
                env=env,
                device='auto',
                tensorboard_log=LOG_DIR
            )
        else:
            model = PPO.load(
                CHECKPOINT_PATH,
                env=env,
                device='auto',
                tensorboard_log=LOG_DIR
            )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Current timesteps: {model.num_timesteps:,}")
        
        # === UPDATE HYPERPARAMETERS FOR STABILITY ===
        print("\nUpdating hyperparameters for recovery...")
        model.learning_rate = LEARNING_RATE
        model.ent_coef = ENT_COEF
        model.n_steps = N_STEPS
        model.batch_size = BATCH_SIZE
        model.gamma = GAMMA
        model.gae_lambda = GAE_LAMBDA
        
        print("✓ Hyperparameters updated")
        
    else:
        print(f"\n⚠ WARNING: Checkpoint not found: {CHECKPOINT_PATH}")
        print("Creating new model from scratch...")
        
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
    
    # === SETUP CALLBACKS ===
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,  # Adjust for parallel envs
        save_path=OUTPUT_DIR,
        name_prefix="ppo_recovery",
        verbose=1
    )
    
    # === TRAINING ===
    print("\n" + "="*60)
    print("STARTING RECOVERY TRAINING")
    print("="*60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoint frequency: {CHECKPOINT_FREQ:,} steps")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="ppo_recovery",
            callback=checkpoint_callback,
            reset_num_timesteps=False  # Continue from checkpoint
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
        print("1. Evaluate the recovery model:")
        print(f"   python evaluation/evaluate_recovery.py")
        print("2. Compare with 150k baseline")
        print("3. Check P2P trading activity (should be > 0)")
        print("4. Monitor SoC stability (should stay > 20%)")
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
