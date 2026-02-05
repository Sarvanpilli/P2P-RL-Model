# eval_per_agent.py
# Evaluates a saved PPO model (EnergyMarketEnvRobust) and writes per-episode + per-agent kWh stats + SOC.
import os
import sys
import argparse
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.energy_env_robust import EnergyMarketEnvRobust

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models_phase2/ppo_advanced_final.zip", help="Path to PPO .zip")
    p.add_argument("--data", default="test_day_profile.csv", help="CSV for demand/PV replay")
    p.add_argument("--episodes", type=int, default=int(os.environ.get("EVAL_EPISODES", "50")))
    p.add_argument("--ep-length", type=int, default=int(os.environ.get("EVAL_EP_LENGTH", "240")))
    p.add_argument("--out-csv", default=None, help="Output CSV (default: same dir as model)")
    return p.parse_args()

args = parse_args()
MODEL_PATH = os.path.abspath(args.model)
N_EPISODES = args.episodes
EP_LENGTH = args.ep_length
OUT_CSV = args.out_csv or os.path.join(os.path.dirname(MODEL_PATH), "eval_per_agent.csv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found: " + MODEL_PATH)

# Build env to infer n_agents and spaces (must match saved model)
def make_env():
    return EnergyMarketEnvRobust(
        n_agents=4,
        data_file=args.data,
        random_start_day=True,
        forecast_horizon=1,
        seed=42,
    )
env = DummyVecEnv([make_env])
vecnorm_path = os.path.join(os.path.dirname(MODEL_PATH), "vec_normalize.pkl")
if os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

print("Loading model:", MODEL_PATH)
model = PPO.load(MODEL_PATH, env=env)
inner = env.venv.envs[0] if hasattr(env, "venv") else env.envs[0]
N_AGENTS = inner.n_agents
timestep_hours = getattr(inner, "timestep_hours", 1.0)
obs_dim = inner.observation_space.shape[0]
per_agent_obs = obs_dim // N_AGENTS  # Robust obs: [dem, soc, pv, ...] -> soc at index 1

rows = []
soc_time_series_all = []

for ep in range(N_EPISODES):
    obs = env.reset()
    soc_series = []
    ep_export_kwh = np.zeros(N_AGENTS)
    ep_import_kwh = np.zeros(N_AGENTS)
    ep_batt_throughput = np.zeros(N_AGENTS)
    ep_reward = 0.0
    ep_overload = 0

    for t in range(EP_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        r = float(np.array(rewards).reshape(-1)[0])
        ep_reward += r
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        trades_kw = np.array(info.get("trades_kw", np.zeros(N_AGENTS)), dtype=float)
        exports_kw = np.clip(trades_kw, 0, None)
        imports_kw = np.abs(np.clip(trades_kw, None, 0))
        ep_export_kwh += exports_kw * timestep_hours
        ep_import_kwh += imports_kw * timestep_hours
        ep_batt_throughput += np.array(info.get("battery_throughput_delta_kwh", np.zeros(N_AGENTS)))
        ep_overload += 1 if (info.get("line_overload_kw", 0.0) > 1e-6) else 0

        flat_obs = np.array(obs).reshape(-1)
        per = flat_obs.reshape(N_AGENTS, per_agent_obs)
        socs = per[:, 1]
        soc_series.append(socs.copy())

        if (isinstance(dones, (list, tuple)) and dones[0]) or (isinstance(dones, bool) and dones):
            break

    soc_time_series_all.append(np.vstack(soc_series) if soc_series else np.zeros((1, N_AGENTS)))

    rows.append({
        "episode": ep,
        "total_reward": float(ep_reward),
        "total_export_kwh": float(ep_export_kwh.sum()),
        "total_import_kwh": float(ep_import_kwh.sum()),
        "per_agent_export_kwh": ";".join([f"{v:.3f}" for v in ep_export_kwh]),
        "per_agent_import_kwh": ";".join([f"{v:.3f}" for v in ep_import_kwh]),
        "battery_throughput_kwh": float(ep_batt_throughput.sum()),
        "overload_events": int(ep_overload),
    })

# write CSV
fieldnames = list(rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# additionally save a simple SOC time series file (numpy npz) for plotting
soc_npz = os.path.join(os.path.dirname(MODEL_PATH), "eval_soc_time_series.npz")
np.savez(soc_npz, *soc_time_series_all)
print("Saved per-episode CSV:", OUT_CSV)
print("Saved SOC time series (npz):", soc_npz)

# print summary
rewards = np.array([r["total_reward"] for r in rows])
exports = np.array([r["total_export_kwh"] for r in rows])
imports = np.array([r["total_import_kwh"] for r in rows])
print("=== Summary ===")
print("N episodes:", N_EPISODES)
print(f"Avg total reward: {rewards.mean():.2f}  std: {rewards.std():.2f}")
print(f"Avg export (kWh/episode): {exports.mean():.2f}")
print(f"Avg import (kWh/episode): {imports.mean():.2f}")
