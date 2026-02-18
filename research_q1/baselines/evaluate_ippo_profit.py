
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train.energy_env_recovery import EnergyMarketEnvRecovery
from research_q1.data.data_loader import ResearchDataLoader

def evaluate_ippo_profit(
    model_path="research_q1/models/ippo_baseline/ippo_final",
    vec_norm_path="research_q1/models/ippo_baseline/vec_normalize.pkl",
    date_str="2017-01-01"
):
    print(f"\n Evaluating IPPO Profit for {date_str}...")
    
    # 1. Setup Env
    def make_env():
        return EnergyMarketEnvRecovery(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=True, 
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            forecast_noise_std=0.05,
            diversity_mode=True,
            seed=42
        )
    
    # 2. Load Normalization and Model
    env = DummyVecEnv([make_env]) # Env is vectorized
    try:
        if os.path.exists(vec_norm_path):
             env = VecNormalize.load(vec_norm_path, env)
             env.training = False 
             env.norm_reward = False
             print("✓ Loaded VecNormalize stats")
        else:
             print("⚠ VecNormalize file not found. Results may be inaccurate.")
    except Exception as e:
        print(f"⚠ Could not load VecNormalize: {e}")
    
    # Load Model
    if not os.path.exists(model_path + ".zip"):
         checkpoints = [f for f in os.listdir("research_q1/models/ippo_baseline") if f.endswith(".zip")]
         if checkpoints:
             # Sort by timestep number
             def get_step(name):
                 parts = name.split('_')
                 for p in parts:
                     if p.isdigit(): return int(p)
                 return 0
             checkpoints.sort(key=get_step)
             
             model_path = os.path.join("research_q1/models/ippo_baseline", checkpoints[-1]).replace(".zip", "")
             print(f"✓ Loading checkpoint: {model_path}")
         else:
             print("❌ Error: No model found.")
             return

    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Setup Data Access
    loader = ResearchDataLoader("processed_hybrid_data.csv")
    full_df = loader.load_data()
    prices_buy, prices_sell = loader.get_market_prices()
    
    # Find start index
    try:
        start_idx = full_df.index.get_loc(date_str)
        if isinstance(start_idx, slice): start_idx = start_idx.start
        elif hasattr(start_idx, '__iter__'): start_idx = start_idx[0]
        print(f"✓ Date {date_str} starts at index {start_idx}")
    except KeyError:
        print(f"⚠ Date {date_str} not found. Using index 0.")
        start_idx = 0
        
    # 4. Force Env State
    # Note: DummyVecEnv wraps envs in `envs` list
    raw_env = env.envs[0]
    raw_env.reset() 
    
    # Overwrite indices
    raw_env.current_idx = start_idx
    raw_env.day_start_idx = start_idx
    raw_env.timestep_count = 0
    
    # Refresh Observation
    raw_obs = raw_env._get_obs(0.0, 0.0)
    obs = env.normalize_obs(raw_obs) if isinstance(env, VecNormalize) else raw_obs
    
    # 5. Simulation Loop
    total_profit = 0.0
    horizon = 24
    
    print("-" * 65)
    print(f"{'Hour':<5} | {'Net Load (kW)':<15} | {'Revenue ($)':<12} | {'Cost ($)':<12} | {'Profit ($)':<12}")
    print("-" * 65)
    
    for t in range(horizon):
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        # FIX: Ensure action is batched for VecEnv
        # If action is 1D (unbatched), wrap it
        if len(action.shape) == 1:
            step_action = [action]
        else:
            step_action = action
            
        # Step
        obs, rewards, dones, infos = env.step(step_action)
        
        # Note: obs returned by step is already normalized and batched (1, obs_dim)
        # We can pass it directly to predict next time.
        # But wait, `obs` variable needs to be updated.
        # However, `model.predict` handles batched input too.
        # If we pass batched obs (1, dim), predict returns batched action (1, dim) or (1,)?
        # SB3 predict: if input is vectorized, output is vectorized.
        # So loop should be stable.
        
        # Calculate Financial Profit
        step_profit = 0.0
        step_net_load = 0.0
        step_revenue = 0.0
        step_cost = 0.0
        
        p_buy = prices_buy[t]
        p_sell = prices_sell[t]
        
        data_idx = start_idx + t
        row = raw_env.df.iloc[data_idx]
        
        for i in range(4):
            node = raw_env.nodes[i]
            
            # 1. Base Data
            if i == 0:
                 d = row.get('agent_0_demand', 0.0); g = row.get('agent_0_pv', 0.0)
            elif i == 1:
                 d = row.get('agent_1_demand', 0.0); g = row.get('agent_1_wind', 0.0)
            elif i == 2:
                 d = row.get('agent_2_demand', 0.0); g = row.get('agent_2_pv', 0.0)
            elif i == 3:
                 d = row.get('agent_3_demand', 0.0); g = row.get('agent_3_pv', 0.0)
            
            # 2. Battery Delta
            p_batt = node.last_power_kw 
            
            # 3. Net Load
            net_load = d - g + p_batt
            step_net_load += net_load
            
            # 4. Grid Interaction
            if net_load > 0: # Import
                c = net_load * p_buy
                step_cost += c
                step_profit -= c
            else: # Export
                r = abs(net_load) * p_sell
                step_revenue += r
                step_profit += r
                
        total_profit += step_profit
        print(f"{t:<5} | {step_net_load:<15.2f} | {step_revenue:<12.2f} | {step_cost:<12.2f} | {step_profit:<12.2f}")
        
    print("-" * 65)
    print(f"TOTAL IPPO PROFIT: ${total_profit:.2f}")
    print("-" * 65)

if __name__ == "__main__":
    evaluate_ippo_profit()
