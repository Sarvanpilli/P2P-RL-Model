# eval_metrics.py
# Robust batch evaluation that builds an env matching the saved model's spaces,
# runs a number of episodes and writes summary CSV.

import os
import zipfile
import pickle
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from energy_env_improved import EnergyMarketEnv

# === CONFIG (edit if needed) ===
MODEL_PATH = r"F:\Projects\P2P-RL-Model\models\ppo_energy_continued_parallel.zip"
N_EPISODES = int(os.environ.get("EVAL_EPISODES", "50"))
EP_LENGTH = int(os.environ.get("EVAL_EP_LENGTH", "240"))
OUTPUT_CSV = os.path.join(os.path.dirname(MODEL_PATH), "eval_metrics_summary.csv")
N_AGENTS = 4  # change if your model used a different number of agents
# ================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

print("Using model:", MODEL_PATH)

# 1) Try to extract observation/action spaces from data.pkl inside the zip
saved_obs_space = None
saved_act_space = None
try:
    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        if "data.pkl" in archive.namelist():
            with archive.open("data.pkl", "r") as f:
                data = pickle.load(f)
                saved_obs_space = data.get("observation_space", None)
                saved_act_space = data.get("action_space", None)
                print("Extracted spaces from data.pkl in model zip.")
        else:
            print("data.pkl not found inside zip; will try PPO.load fallback.")
except Exception as e:
    print("Error reading model zip:", e)

# 2) Fallback: use PPO.load without env to extract spaces if necessary
if saved_obs_space is None or saved_act_space is None:
    try:
        tmp = PPO.load(MODEL_PATH, print_system_info=False)
        saved_obs_space = saved_obs_space or getattr(tmp, "observation_space", None)
        saved_act_space = saved_act_space or getattr(tmp, "action_space", None)
        print("Extracted spaces via PPO.load fallback.")
    except Exception as e:
        print("PPO.load fallback failed:", e)

if saved_obs_space is None or saved_act_space is None:
    raise RuntimeError("Could not determine saved observation/action spaces from model. Aborting.")

print("Saved observation space shape:", saved_obs_space.shape)
print("Saved action space shape:", saved_act_space.shape)

# 3) Compute obs_size and infer per-agent layout and forecast_horizon
obs_size = int(np.prod(saved_obs_space.shape))
if obs_size % N_AGENTS != 0:
    raise ValueError(f"Saved obs_size {obs_size} not divisible by N_AGENTS={N_AGENTS}. Adjust N_AGENTS if needed.")

per_agent_len = obs_size // N_AGENTS
# per_agent_len = 3 (demand,soc,pv) + 2 (total_export,total_import) + 2*forecast_horizon
forecast_horizon = int((per_agent_len - 5) / 2)
if forecast_horizon < 0:
    forecast_horizon = 0

print(f"Per-agent length: {per_agent_len}, inferred forecast_horizon: {forecast_horizon}")

# 4) Infer battery_capacity_kwh if available (soc index == 1 in per-agent block)
battery_capacity_guess = None
try:
    high = np.array(saved_obs_space.high).reshape(-1)
    per_high = high.reshape(N_AGENTS, per_agent_len)
    soc_high = float(per_high[0, 1])
    if np.isfinite(soc_high) and 0 < soc_high < 1e6:
        battery_capacity_guess = soc_high
        print("Inferred battery_capacity_kwh from saved obs_high:", battery_capacity_guess)
except Exception:
    battery_capacity_guess = None

if battery_capacity_guess is None:
    battery_capacity_guess = 50.0
    print("Falling back to battery_capacity_kwh =", battery_capacity_guess)

# 5) Infer action highs (battery_max_charge_kw and per_agent_max_kw) from saved action_space.high
battery_max_charge_kw_guess = None
per_agent_max_kw_guess = None
try:
    act_high = np.array(saved_act_space.high).reshape(-1)
    per_act_len = act_high.shape[0] // N_AGENTS
    per_act = act_high.reshape(N_AGENTS, per_act_len)
    battery_max_charge_kw_guess = float(per_act[0, 0])
    per_agent_max_kw_guess = float(per_act[0, 1])
    print("Inferred battery_max_charge_kw:", battery_max_charge_kw_guess,
          "per_agent_max_kw:", per_agent_max_kw_guess)
except Exception as e:
    print("Could not infer action highs; using defaults. Error:", e)
    battery_max_charge_kw_guess = 25.0
    per_agent_max_kw_guess = 120.0

# 6) Build env factory using the inferred parameters EXACTLY
def make_env():
    return EnergyMarketEnv(
        n_agents=N_AGENTS,
        forecast_horizon=forecast_horizon,
        battery_capacity_kwh=battery_capacity_guess,
        battery_max_charge_kw=battery_max_charge_kw_guess,
        per_agent_max_kw=per_agent_max_kw_guess,
        shaping_coef=float(os.environ.get("SHAPING_COEF", "0.0")),
        seed=None
    )

env = DummyVecEnv([make_env])

print("Created environment with parameters inferred from model:")
print(f"  forecast_horizon={forecast_horizon}, battery_capacity_kwh={battery_capacity_guess}, battery_max_charge_kw={battery_max_charge_kw_guess}, per_agent_max_kw={per_agent_max_kw_guess}")

# 7) Load model with matching env (this will validate exact match)
model = PPO.load(MODEL_PATH, env=env)
print("Model loaded successfully.")

# 8) Evaluate N_EPISODES and collect KPIs
rows = []
for ep in range(N_EPISODES):
    obs = env.reset()
    total_reward = 0.0
    total_export = 0.0
    total_import = 0.0
    battery_throughput = 0.0
    curtail_events = 0
    overload_events = 0

    for t in range(EP_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        reward = float(np.array(rewards).reshape(-1)[0])
        total_reward += reward

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        total_export += float(info.get("total_export_kw_after", 0.0))
        total_import += float(info.get("total_import_kw_after", 0.0))
        battery_throughput += float(np.sum(info.get("battery_throughput_kwh", np.zeros(N_AGENTS))))
        curtail_events += int(np.sum(info.get("curtailment_kw", np.zeros(N_AGENTS)) > 1e-6))
        overload_events += int(1 if info.get("line_overload_kw", 0.0) > 1e-6 else 0)

        # handle both vec-env done formats
        if isinstance(dones, (list, tuple)) and dones[0]:
            break
        if isinstance(dones, bool) and dones:
            break

    rows.append({
        "episode": ep,
        "total_reward": total_reward,
        "total_export_kw": total_export,
        "total_import_kw": total_import,
        "battery_throughput_kwh": battery_throughput,
        "curtail_events": curtail_events,
        "overload_events": overload_events
    })
    print(f"Ep {ep:03d} reward={total_reward:.2f} export={total_export:.2f} import={total_import:.2f} curtail={curtail_events} overload={overload_events}")

# 9) Write CSV summary
fieldnames = ["episode","total_reward","total_export_kw","total_import_kw","battery_throughput_kwh","curtail_events","overload_events"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# 10) Print aggregated stats
rewards = np.array([r["total_reward"] for r in rows])
exports = np.array([r["total_export_kw"] for r in rows])
imports = np.array([r["total_import_kw"] for r in rows])
print("\n=== Summary ===")
print(f"N episodes: {N_EPISODES}")
print(f"Avg total reward: {rewards.mean():.2f}  std: {rewards.std():.2f}")
print(f"Avg total export (per episode): {exports.mean():.2f}")
print(f"Avg total import (per episode): {imports.mean():.2f}")
print(f"CSV saved to: {OUTPUT_CSV}")
