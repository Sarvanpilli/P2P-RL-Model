"""debug_market.py — Diagnose zero P2P volume from SLIM model."""
import sys, os, numpy as np
sys.path.insert(0, os.path.abspath("."))

from train.energy_env_robust import EnergyMarketEnvRobust
from stable_baselines3 import PPO

model_path = "research_q1/models/slim_ppo/slim_ppo_250000_steps.zip"

env = EnergyMarketEnvRobust(
    n_agents=4,
    data_file="processed_hybrid_data.csv",
    random_start_day=False,
    enable_predictive_obs=True,
    forecast_horizon=4,
    diversity_mode=True,
    forecast_noise_std=0.05,
)

try:
    model = PPO.load(model_path, env=env)
    print(f"Model loaded OK. Obs space: {model.observation_space.shape}")
    print(f"  Policy type: {type(model.policy).__name__}")
except Exception as e:
    print(f"Load FAILED: {e}")
    # Try with matching obs dim
    print("\nAttempting to reload without env (custom_objects)...")
    try:
        model = PPO.load(model_path)
        print(f"Loaded without env. Obs space: {model.observation_space.shape}")
    except Exception as e2:
        print(f"Also failed: {e2}")
    sys.exit(1)

obs, _ = env.reset(seed=0)
p2p_steps = 0
grid_steps = 0
total_p2p = 0.0

print(f"\n{'t':>4}  {'trade[0]':>9} {'trade[1]':>9} {'trade[2]':>9} {'trade[3]':>9}  {'p2p':>7} {'grid':>7} {'mkt_$':>6}")
print("-"*80)

for t in range(168):  # 1 week
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    p2p  = info.get('p2p_volume_kwh_step', info.get('p2p_volume', 0.0))
    grid = info.get('total_import', info.get('grid_import', 0.0))
    mkt  = info.get('market_price', 0.0)

    if p2p > 0:
        p2p_steps += 1
        total_p2p += p2p
    if grid > 0:
        grid_steps += 1

    if t < 24:
        raw = action.reshape(4, 3)
        trades = " ".join(f"{raw[i,1]:+.2f}" for i in range(4))
        print(f"{t:4d}  {trades}  {p2p:7.3f} {grid:7.3f} {mkt:6.3f}")

    if done or truncated:
        obs, _ = env.reset(seed=t)

print(f"\n{'='*60}")
print(f"Summary over 168 steps:")
print(f"  Steps with P2P trades:  {p2p_steps}/168")
print(f"  Total P2P volume:       {total_p2p:.4f} kWh")
print(f"  Steps with grid import: {grid_steps}/168")

# Also print what info keys exist
env.reset(seed=99)
act = env.action_space.sample()
_, _, _, _, info_sample = env.step(act)
print(f"\nInfo dict keys: {list(info_sample.keys())}")
env.close()
