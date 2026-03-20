"""debug_obs_space.py — Diagnose observation space mismatches."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("."))

from train.energy_env_robust import EnergyMarketEnvRobust

configs_to_test = [
    {"forecast_horizon": 4, "n_agents": 4, "label": "horizon=4 (current)"},
    {"forecast_horizon": 5, "n_agents": 4, "label": "horizon=5"},
    {"forecast_horizon": 6, "n_agents": 4, "label": "horizon=6"},
    {"forecast_horizon": 4, "n_agents": 4, "label": "horizon=4 diversity_mode=False", "diversity_mode": False},
]

print("=== Environment Obs Space Tests ===")
for cfg in configs_to_test:
    try:
        env = EnergyMarketEnvRobust(
            n_agents=cfg.get("n_agents", 4),
            data_file="processed_hybrid_data.csv",
            random_start_day=False,
            enable_predictive_obs=True,
            forecast_horizon=cfg.get("forecast_horizon", 4),
            diversity_mode=cfg.get("diversity_mode", True),
            forecast_noise_std=0.05,
        )
        print(f"  {cfg['label']:40s}: obs_dim = {env.observation_space.shape[0]}")
        env.close()
    except Exception as e:
        print(f"  {cfg['label']:40s}: ERROR — {e}")

# Inspect saved model files directly
print("\n=== Saved Model Obs Space Inspection ===")
from stable_baselines3 import PPO
import zipfile, json, io, torch

paths_to_check = [
    "research_q1/models/slim_ppo/slim_ppo_250000_steps.zip",
    "research_q1/models/ippo_baseline/ippo_baseline_10000_steps.zip",
    "research_q1/models/slim_ablation_N4_NoSafety/ppo_N4_NoSafety_100000_steps.zip",
]
for path in paths_to_check:
    if not os.path.exists(path):
        print(f"\n  {path}: FILE NOT FOUND")
        continue
    try:
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            print(f"\n  {path}")
            print(f"    Contents: {names}")
            if 'policy.json' in names:
                data = json.loads(z.read('policy.json'))
                print(f"    obs_space from policy.json: {data.get('observation_space', 'not found')}")
            if 'data' in names:
                buf = io.BytesIO(z.read('data'))
                d = torch.load(buf, map_location='cpu')
                obs_shape = d.get('observation_space', {})
                print(f"    obs_space from data: {obs_shape}")
    except Exception as e:
        print(f"  {path}: cannot inspect — {e}")
