# train_sb3_ppo.py
"""
Train a shared PPO policy (Stable-Baselines3) controlling all agents in the
energy_env_improved.Environment (flattened obs/actions).

Saves a best model to ./models/ppo_energy.zip and logs to ./logs/.
Run:
    python train_sb3_ppo.py
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# import your env - handle both running from root and from train directory
try:
    from train.energy_env_improved import EnergyMarketEnv
except ImportError:
    # If running from train directory, use relative import
    from energy_env_improved import EnergyMarketEnv

# ==== CONFIG ====
ENV_NAME = "EnergyMarket-v0"
TIMESTEPS = 200_000          # start here; scale up to millions later
N_AGENTS = 4
SEED = 42
MODEL_DIR = "models"
LOG_DIR = "logs"
EVAL_EPISODES = 6
EVAL_FREQ = 10_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==== Environment factory ====
def make_env():
    def _init():
        env = EnergyMarketEnv(n_agents=N_AGENTS,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=120.0,
                              base_price=0.12,
                              price_slope=0.002,
                              overload_multiplier=25.0,
                              seed=np.random.randint(0, 2**31 - 1))
        # wrap with Monitor for SB3 logging
        env = Monitor(env)
        return env
    return _init

# Create a vectorized environment
vec_env = DummyVecEnv([make_env() for _ in range(1)])  # you can increase to multiple env copies
# Normalize observations and rewards (recommended for PPO)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Policy architecture: small MLP to handle flattened obs/actions
policy_kwargs = dict(net_arch=dict(pi=[512, 256], vf=[512, 256]))

# Callbacks: checkpoint + eval
checkpoint_callback = CheckpointCallback(save_freq=50_00, save_path=MODEL_DIR,
                                         name_prefix="ppo_energy_checkpoint")
# Use EvalCallback to evaluate on separate envs
eval_env = DummyVecEnv([make_env() for _ in range(1)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             n_eval_episodes=EVAL_EPISODES, deterministic=True, render=False)

# ==== Create model ====
model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            seed=SEED,
            batch_size=2048,
            n_steps=2048,
            learning_rate=3e-4,
            ent_coef=0.001,
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR)

# ==== Start training ====
print("Starting training for", TIMESTEPS, "timesteps...")
model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint_callback, eval_callback])

# Save final model (and the VecNormalize stats)
model.save(os.path.join(MODEL_DIR, "ppo_energy_final"))
vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

print("Training completed. Models saved in", MODEL_DIR)
