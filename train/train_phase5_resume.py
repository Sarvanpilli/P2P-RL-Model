"""
Phase 5 Hybrid Training - Resume Script (Phased Approach)

This script resumes training from an existing checkpoint with a phased approach:
- Phase 1: 50k → 200k steps (150k additional)
- Phase 2: 200k → 350k steps (150k additional) - Run after cooldown
- Phase 3: 350k → 500k steps (150k additional) - Run after cooldown

Usage:
    python train/train_phase5_resume.py --phase 1
    python train/train_phase5_resume.py --phase 2
    python train/train_phase5_resume.py --phase 3
"""

import os
import sys
import argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
    print("Using RecurrentPPO (LSTM) Policy")
except ImportError:
    USING_RECURRENT = False
    print("WARNING: sb3-contrib not found. Falling back to PPO.")
    from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# Training phases configuration
PHASES = {
    1: {"start": 50000, "end": 200000, "checkpoint": "ppo_hybrid_50000_steps"},
    2: {"start": 200000, "end": 350000, "checkpoint": "ppo_hybrid_200000_steps"},
    3: {"start": 350000, "end": 500000, "checkpoint": "ppo_hybrid_350000_steps"}
}

def make_env():
    """Create Phase 5 Hybrid Environment"""
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4,
        enable_predictive_obs=True,
        forecast_noise_std=0.05,
        diversity_mode=True
    )
    return env

def resume_training(phase: int):
    """Resume training for specified phase"""
    
    if phase not in PHASES:
        raise ValueError(f"Invalid phase {phase}. Must be 1, 2, or 3.")
    
    phase_config = PHASES[phase]
    start_steps = phase_config["start"]
    end_steps = phase_config["end"]
    checkpoint_name = phase_config["checkpoint"]
    
    additional_steps = end_steps - start_steps
    
    print(f"\n{'='*60}")
    print(f"PHASE {phase} TRAINING")
    print(f"{'='*60}")
    print(f"Resuming from: {start_steps:,} steps")
    print(f"Target: {end_steps:,} steps")
    print(f"Additional training: {additional_steps:,} steps")
    print(f"Checkpoint: {checkpoint_name}.zip")
    print(f"{'='*60}\n")
    
    model_dir = "models_phase5_hybrid"
    log_dir = "./tensorboard_logs/"
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_name}.zip")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please ensure Phase {phase-1} training completed successfully."
        )
    
    # Create environment
    n_envs = 1  # Debug Mode
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Try to load VecNormalize stats if they exist
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
    else:
        print("No VecNormalize stats found, creating new normalization wrapper")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    if USING_RECURRENT:
        model = RecurrentPPO.load(
            checkpoint_path,
            env=env,
            device='auto',
            tensorboard_log=log_dir
        )
    else:
        model = PPO.load(
            checkpoint_path,
            env=env,
            device='auto',
            tensorboard_log=log_dir
        )
    
    print(f"Model loaded successfully!")
    print(f"Current timesteps: {model.num_timesteps:,}")
    
    # Setup checkpoint callback for this phase
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=model_dir,
        name_prefix=f"ppo_hybrid",
        verbose=1
    )
    
    # Train
    print(f"\nStarting Phase {phase} training...")
    print(f"Training for {additional_steps:,} additional steps...\n")
    
    model.learn(
        total_timesteps=additional_steps,
        tb_log_name=f"ppo_hybrid_phase{phase}",
        callback=checkpoint_callback,
        reset_num_timesteps=False  # Continue counting from checkpoint
    )
    
    # Save final model for this phase
    final_model_name = f"ppo_hybrid_{end_steps}_steps"
    model.save(os.path.join(model_dir, final_model_name))
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    print(f"\n{'='*60}")
    print(f"PHASE {phase} COMPLETE!")
    print(f"{'='*60}")
    print(f"Final model saved: {final_model_name}.zip")
    print(f"Total timesteps: {model.num_timesteps:,}")
    print(f"{'='*60}\n")
    
    if phase < 3:
        print(f"⚠️  COOLDOWN RECOMMENDED ⚠️")
        print(f"Allow system to cool down before starting Phase {phase+1}")
        print(f"Next command: python train/train_phase5_resume.py --phase {phase+1}\n")
    else:
        print(f"🎉 ALL PHASES COMPLETE! 🎉")
        print(f"Final model: {final_model_name}.zip")
        print(f"Total training: 500,000 steps\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Phase 5 Hybrid Training")
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Training phase: 1 (50k→200k), 2 (200k→350k), 3 (350k→500k)"
    )
    
    args = parser.parse_args()
    resume_training(args.phase)
