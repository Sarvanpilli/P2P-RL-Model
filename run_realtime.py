import time
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from realtime_env_wrapper import RealTimeEnvWrapper

# 1. Load Data
DATA_FILE = "test_day_profile.csv"
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} timesteps.")

# 2. Setup Environment
N_AGENTS = 4
def make_env():
    # Same config as training
    return RealTimeEnvWrapper(dataframe=df,
                              n_agents=N_AGENTS,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=120.0,
                              base_price=0.12,
                              price_slope=0.002,
                              overload_multiplier=25.0,
                              forecast_horizon=0) # Assuming 0 for now as per debug check

# Vectorize
vec_env = DummyVecEnv([make_env])

# 3. Load Model & Normalization Stats
models_dir = Path("models")
model_path = models_dir / "ppo_energy_final.zip"
if not model_path.exists():
    # Fallback to checkpoint
    checkpoints = list(models_dir.glob("ppo_energy_checkpoint_*.zip"))
    if checkpoints:
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = checkpoints[0]

print(f"Loading model from {model_path}...")
vecnorm_path = models_dir / "vec_normalize.pkl"

if vecnorm_path.exists():
    print("Loading normalization stats...")
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
else:
    print("WARNING: No normalization stats found. Model performance may be degraded.")

model = PPO.load(str(model_path), env=vec_env)

# 4. Run "Real-Time" Loop
print("\nStarting Real-Time Simulation...")
print("---------------------------------")

obs = vec_env.reset()
done = False

# We'll simulate 1 hour every 1 second (or faster for demo)
SLEEP_TIME = 1.0 

try:
    while not done:
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, rewards, dones, infos = vec_env.step(action)
        done = dones[0]
        info = infos[0]
        
        # Get raw state from the wrapper (accessed via the vec_env)
        # vec_env.envs[0] is the RealTimeEnvWrapper instance
        env_instance = vec_env.envs[0]
        current_step = env_instance.current_step_idx
        state = env_instance.state
        
        # --- DASHBOARD ---
        print(f"\n=== Hour {current_step}/{len(df)} ===")
        print(f"Market Price: ${info['market_price']:.3f}/kWh")
        print(f"Grid Status : Export={info['total_export_kw_after']:.1f} kW | Import={info['total_import_kw_after']:.1f} kW")
        if info['line_overload_kw'] > 0:
            print(f"WARNING     : LINE OVERLOAD! ({info['line_overload_kw']:.1f} kW)")
        
        print("-" * 60)
        print(f"{'Agent':<6} | {'Demand (kW)':<12} | {'PV (kW)':<10} | {'SOC (%)':<10} | {'Action (Batt/Grid)':<20}")
        print("-" * 60)
        
        # Reconstruct actions from the info or state if possible, 
        # but here we can just read the state. 
        # The 'intended_injection_kw' is in info.
        
        for i in range(N_AGENTS):
            dem = state[i, 0]
            pv = state[i, 2]
            soc = state[i, 1]
            cap = env_instance.battery_capacity_kwh
            soc_pct = (soc / cap) * 100
            
            # We don't have the exact battery action in 'info' easily broken down per agent 
            # without modifying the env to pass it, but we can infer or just show grid trade.
            grid_trade = info['intended_injection_kw'][i]
            
            print(f"A{i:<5} | {dem:<12.2f} | {pv:<10.2f} | {soc_pct:<10.1f} | Grid: {grid_trade:+.2f} kW")
            
        print("-" * 60)
        print(f"Step Reward: {rewards[0]:.3f}")
        
        if done:
            print("\nEnd of Data.")
            break
            
        time.sleep(SLEEP_TIME)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
