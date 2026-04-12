
import os
import sys
import numpy as np
import pandas as pd
import json

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def run_baseline():
    print("Running Carbon Baseline Simulation (Grid-Only)...")
    
    # 1. Setup Env
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=False
    )
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    total_co2 = 0.0
    total_profit = 0.0
    
    history = []
    
    while not (done or truncated):
        # GRID ONLY POLICY:
        # action[:, 0] = 0 (No battery)
        # action[:, 1] = 0 (No trade intent)
        # action[:, 2] = 0 (Price 0.15)
        action = np.zeros((4, 3))
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Grid price and CO2 intensity from data
        # Clean profit: - (grid_import * retail_price)
        # CO2: grid_import * carbon_intensity
        
        # Total import is in 'info'
        grid_import = info.get('total_import', 0)
        co2_intensity = info.get('co2_intensity', 0.5) # kg/kWh approx
        retail_p = info.get('grid_buy_price', 0.30)
        
        step_co2 = grid_import * co2_intensity
        step_cost = grid_import * retail_p
        
        total_co2 += step_co2
        total_profit -= step_cost
        
        history.append({
            't': env.timestep_count,
            'co2': step_co2,
            'profit': -step_cost
        })
        
    baseline_stats = {
        'total_co2_kg': float(total_co2),
        'total_profit_usd': float(total_profit),
        'avg_co2_per_step': float(total_co2 / len(history))
    }
    
    os.makedirs("research_q1/results", exist_ok=True)
    with open("research_q1/results/carbon_baseline.json", "w") as f:
        json.dump(baseline_stats, f, indent=4)
        
    print(f"Baseline saved: {baseline_stats['total_co2_kg']:.2f} kg CO2, {baseline_stats['total_profit_usd']:.2f} USD")

if __name__ == "__main__":
    run_baseline()
