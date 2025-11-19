# debug_eval_verbose.py
# Robust loader: reads saved obs & action spaces from model zip (or PPO.load fallback),
# builds a matching EnergyMarketEnv and runs a verbose unnormalized evaluation.

import os
import zipfile
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from energy_env_improved import EnergyMarketEnv

# Path to your saved model zip
MODEL_PATH = r"F:\Projects\P2P-RL-Model\models\ppo_energy_continued_parallel.zip"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

print("Inspecting model file:", MODEL_PATH)

saved_obs_space = None
saved_act_space = None

# Try reading data.pkl inside the zip (preferred)
try:
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        if "data.pkl" in archive.namelist():
            with archive.open("data.pkl", "r") as f:
                data = pickle.load(f)
                saved_obs_space = data.get("observation_space", None)
                saved_act_space = data.get("action_space", None)
                print("Found data.pkl inside zip and extracted spaces.")
        else:
            print("data.pkl not found inside zip; will try PPO.load fallback.")
except Exception as e:
    print("Error reading zip/data.pkl:", e)

# Fallback: use PPO.load to extract spaces if necessary
if saved_obs_space is None or saved_act_space is None:
    try:
        tmp = PPO.load(MODEL_PATH, print_system_info=False)
        saved_obs_space = getattr(tmp, "observation_space", saved_obs_space)
        saved_act_space = getattr(tmp, "action_space", saved_act_space)
        print("Extracted spaces via PPO.load fallback.")
    except Exception as e:
        print("PPO.load fallback failed:", e)

if saved_obs_space is None or saved_act_space is None:
    raise RuntimeError("Could not determine saved observation/action spaces from model.")

print("Saved observation space shape:", saved_obs_space.shape)
print("Saved action space shape:", saved_act_space.shape)

# Infer per-agent layout and forecast horizon
obs_size = int(np.prod(saved_obs_space.shape))
N_AGENTS = 4  # adjust if your model used different number of agents
per_agent_len = obs_size // N_AGENTS
if per_agent_len * N_AGENTS != obs_size:
    raise ValueError("Saved obs_size not divisible by N_AGENTS; adjust N_AGENTS or inspect saved model.")

forecast_horizon = int((per_agent_len - 5) / 2)
if forecast_horizon < 0:
    forecast_horizon = 0

print(f"Inferred per-agent length: {per_agent_len}, forecast_horizon: {forecast_horizon}")

# Infer battery_capacity_kwh from saved_obs_space.high if available (soc index = 1)
battery_capacity_guess = None
try:
    high = np.array(saved_obs_space.high).reshape(-1)
    per_high = high.reshape(N_AGENTS, per_agent_len)
    soc_high = float(per_high[0, 1])
    if np.isfinite(soc_high) and 0 < soc_high < 1e6:
        battery_capacity_guess = soc_high
        print("Inferred battery_capacity_kwh from obs_high:", battery_capacity_guess)
except Exception:
    battery_capacity_guess = None

if battery_capacity_guess is None:
    battery_capacity_guess = 50.0
    print("Falling back to default battery_capacity_kwh =", battery_capacity_guess)

# Infer battery_max_charge_kw and per_agent_max_kw from saved action_space.high
act_high = np.array(saved_act_space.high).reshape(-1)
try:
    per_act_len = act_high.shape[0] // N_AGENTS
    per_act = act_high.reshape(N_AGENTS, per_act_len)
    battery_max_charge_guess = float(per_act[0, 0])
    per_agent_max_kw_guess = float(per_act[0, 1])
    print("Inferred battery_max_charge_kw:", battery_max_charge_guess,
          "per_agent_max_kw:", per_agent_max_kw_guess)
except Exception as e:
    battery_max_charge_guess = 25.0
    per_agent_max_kw_guess = 120.0
    print("Could not infer action highs, using defaults (25,120). Error:", e)

# Build environment matching saved spaces
def make_env():
    return EnergyMarketEnv(
        n_agents=N_AGENTS,
        forecast_horizon=forecast_horizon,
        battery_capacity_kwh=battery_capacity_guess,
        battery_max_charge_kw=battery_max_charge_guess,
        per_agent_max_kw=per_agent_max_kw_guess,
        shaping_coef=float(os.environ.get("SHAPING_COEF", "1.0")),
        seed=999
    )

env = DummyVecEnv([make_env])

print("Created env with matching spaces. Now loading model with env...")

# Load model with env (this validates spaces match)
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded successfully.")

# Verbose evaluation
obs = env.reset()  # DummyVecEnv.reset() returns observation only
real_env = env.envs[0]
per_len_local = real_env.observation_space.shape[0] // real_env.n_agents

for t in range(240):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

    # rewards is vector-like; get scalar for first env
    reward = float(np.array(rewards).reshape(-1)[0])
    env_info = infos[0] if isinstance(infos, (list, tuple)) else infos

    flat = np.array(obs).reshape(-1)
    per = flat.reshape(real_env.n_agents, per_len_local)

    print(f"\nStep {t}  price={env_info.get('market_price'):.4f} line_ol={env_info.get('line_overload_kw'):.4f} total_reward={reward:.3f}")
    for i in range(real_env.n_agents):
        a_batt = action.reshape(real_env.n_agents, 2)[i, 0]
        a_grid = action.reshape(real_env.n_agents, 2)[i, 1]
        intended = float(env_info.get("intended_injection_kw")[i]) if env_info.get("intended_injection_kw") is not None else 0.0
        curtail = float(env_info.get("curtailment_kw")[i]) if env_info.get("curtailment_kw") is not None else 0.0
        print(f" A{i}: action=[batt {a_batt:.2f}, grid {a_grid:.2f}] soc={per[i,1]:.2f} pv={per[i,2]:.2f} intended={intended:.2f} curtail={curtail:.2f}")

    if dones[0]:
        print("Episode ended at step", t)
        break
