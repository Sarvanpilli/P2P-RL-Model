# resume_parallel.py
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from energy_env_improved import EnergyMarketEnv

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# find latest checkpoint (look for ppo_energy_checkpoint_*.zip or ppo_energy_continued.zip)
candidates = glob.glob(os.path.join(MODEL_DIR, "ppo_energy_checkpoint_*.zip")) + \
             glob.glob(os.path.join(MODEL_DIR, "ppo_energy_continued*.zip")) + \
             glob.glob(os.path.join(MODEL_DIR, "ppo_energy_*.zip"))
if not candidates:
    raise FileNotFoundError("No checkpoint found in models/. Run initial training first.")
# pick the most recently modified
checkpoint = max(candidates, key=os.path.getmtime)
print("Resuming from checkpoint:", checkpoint)

VECSTAT = os.path.join(MODEL_DIR, "vec_normalize.pkl")

N_PARALLEL = 4
N_AGENTS = 4
SEED = 123
CONTINUE_TIMESTEPS = 1_000_000  # change if you want more or less
N_STEPS = 1024

def make_env(seed_offset=0):
    def _init():
        env = EnergyMarketEnv(n_agents=N_AGENTS,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=120.0,
                              base_price=0.12,
                              price_slope=0.01,
                              overload_multiplier=50.0,
                              seed=SEED + seed_offset)
        return Monitor(env)
    return _init

print(f"Creating SubprocVecEnv with {N_PARALLEL} processes...")
env_fns = [make_env(i) for i in range(N_PARALLEL)]
vec_env = SubprocVecEnv(env_fns)

if os.path.exists(VECSTAT):
    print("Loading VecNormalize stats...")
    vec_env = VecNormalize.load(VECSTAT, vec_env)
else:
    print("Creating new VecNormalize wrapper...")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

print("Loading PPO model...")
model = PPO.load(checkpoint, env=vec_env)
# ensure model uses new env
model.set_env(vec_env)
# update hyperparams if desired
model.n_steps = N_STEPS

print("Continuing training for", CONTINUE_TIMESTEPS, "timesteps...")
model.learn(total_timesteps=CONTINUE_TIMESTEPS, reset_num_timesteps=False)

final_path = os.path.join(MODEL_DIR, "ppo_energy_continued_parallel.zip")
print("Saving model to", final_path)
model.save(final_path)
vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
print("Done.")
