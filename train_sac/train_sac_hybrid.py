"""
train_sac_hybrid.py
====================
SAC (Soft Actor-Critic) training for the P2P Energy Trading environment.

Uses the SAME environment (EnergyMarketEnvRobust) and dataset
(processed_hybrid_data.csv) as the PPO Phase-5 Hybrid run so results
are directly comparable.

Key differences vs PPO:
  - Off-policy: SAC uses a replay buffer (default 1M steps) → more sample-efficient
  - Entropy regularisation: automatic temperature tuning (ent_coef="auto")
  - No VecNormalize on observations (SAC normalises internally via batch stats)
    but reward scaling is done manually.

Output:
  models_sac/sac_hybrid_final.zip   ← final weights
  tensorboard_logs/sac_hybrid/      ← TensorBoard scalars
"""

import os
import sys
import argparse

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Allow running from the project root or from the train_sac subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.energy_env_robust import EnergyMarketEnvRobust


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory — mirror of PPO Phase-5 configuration
# ─────────────────────────────────────────────────────────────────────────────
def make_env(data_file: str = "processed_hybrid_data.csv", seed: int = 0):
    """Return a Monitor-wrapped EnergyMarketEnvRobust configured identically to PPO."""
    def _init():
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file=data_file,
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,          # 4-hour lookahead (same as PPO)
            enable_predictive_obs=True,
            forecast_noise_std=0.05,
            diversity_mode=True,         # Hybrid archetypes: Solar/Wind/EV/Standard
        )
        env = Monitor(env)               # Wraps env to log episode stats
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# SAC hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_HYPERPARAMS = dict(
    learning_rate=3e-4,          # Same as PPO default; SAC converges well here
    buffer_size=300_000,         # Replay buffer (stores up to 300k transitions)
    learning_starts=5_000,       # Warm-up random steps before learning begins
    batch_size=256,              # Mini-batch for gradient updates
    tau=0.005,                   # Soft-update coefficient for target networks
    gamma=0.99,                  # Discount factor (same as PPO)
    train_freq=1,                # Update every 1 environment step
    gradient_steps=1,            # Gradient steps per env step (1 = stable)
    ent_coef="auto",             # Automatic entropy tuning (key SAC feature)
    target_update_interval=1,    # Update target network every step
    use_sde=False,               # Standard Normal noise (no state-dependent noise)
    policy_kwargs=dict(
        net_arch=[400, 300],     # Match PPO's MLP [400, 300]
        # SAC policy uses separate actor & critic heads automatically
    ),
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="auto",
)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(
    timesteps: int = 300_000,
    data_file: str = "processed_hybrid_data.csv",
    model_dir: str = "models_sac",
    seed: int = 42,
):
    print("=" * 60)
    print(" SAC Training — P2P Energy Trading (SLIM env)")
    print(f" Dataset       : {data_file}")
    print(f" Total steps   : {timesteps:,}")
    print(f" Seed          : {seed}")
    print(f" Output dir    : {model_dir}")
    print("=" * 60)

    os.makedirs(model_dir, exist_ok=True)

    # Training env (single env; SAC is not easily parallelised due to replay buffer)
    train_env = DummyVecEnv([make_env(data_file, seed)])

    # Separate eval env (no random start so episodes are comparable)
    def make_eval_env():
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file=data_file,
            random_start_day=False,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            enable_predictive_obs=True,
            forecast_noise_std=0.0,      # No noise during evaluation
            diversity_mode=True,
        )
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval_env])

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=model_dir,
        name_prefix="sac_hybrid",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(model_dir, "eval_logs"),
        eval_freq=25_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    # Build SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        **DEFAULT_HYPERPARAMS,
    )
    model.set_random_seed(seed)

    print(f"\n[SAC] Policy network architecture: {DEFAULT_HYPERPARAMS['policy_kwargs']['net_arch']}")
    print(f"[SAC] Observation space : {train_env.observation_space.shape}")
    print(f"[SAC] Action space      : {train_env.action_space.shape}\n")

    # Train
    model.learn(
        total_timesteps=timesteps,
        tb_log_name="sac_hybrid",
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "sac_hybrid_final")
    model.save(final_path)
    print(f"\n[SAC] Training complete. Model saved → {final_path}.zip")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on P2P Energy Trading env")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Total training timesteps (default: 300,000)")
    parser.add_argument("--data_file", type=str, default="processed_hybrid_data.csv",
                        help="Path to dataset CSV (default: processed_hybrid_data.csv)")
    parser.add_argument("--model_dir", type=str, default="models_sac",
                        help="Directory to save SAC checkpoints (default: models_sac)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        data_file=args.data_file,
        model_dir=args.model_dir,
        seed=args.seed,
    )
