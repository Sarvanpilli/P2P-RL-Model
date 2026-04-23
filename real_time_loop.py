
# real_time_loop.py
import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from collections import deque

# Custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train.energy_env_robust import EnergyMarketEnvRobust

STATS_FILE = "live_market_stats.json"
MODEL_PATH = "models_scalable_v5/ppo_n24_v5.zip"



def run_real_time(dataset="hybrid"):
    print(f"Starting SLIM v4 Live Demo on [{dataset.upper()}] dataset...")
    
    data_map = {
        "hybrid": "processed_hybrid_data.csv",
        "ausgrid": "evaluation/ausgrid_p2p_energy_dataset.csv"
    }
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file {MODEL_PATH} not found.")
        return
        
    print(f"Loading SLIM v4 Policy from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    
    def init_simulation(ds_name):
        target_data = data_map.get(ds_name, "processed_hybrid_data.csv")
        new_env = EnergyMarketEnvRobust(
            n_prosumers=24, 
            n_consumers=0, 
            data_file=target_data,
            random_start_day=True,
        )
        new_obs, _ = new_env.reset()
        return new_env, new_obs

    env, obs = init_simulation(dataset)
    
    # Initialize State
    def reset_market_data(ds_type):
        return {
            'dataset_type': ds_type.upper(),
            'total_p2p_volume': 0.0,
            'cumulative_market_profit': 0.0,
            'cumulative_economic_profit': 0.0,
            'rolling_market_profit': 0.0,
            'cumulative_co2': 0.0,
            'cumulative_baseline_co2': 0.0,
            'rolling_grid_dependency': 100.0,
            'cumulative_grid_dependency': 100.0,
            'trade_success_rate': 0.0,
            'behavior_insight': "Initializing simulation...",
            'sim_hour': 0,
            'is_active': True,
            'last_update': "",
            'history': []
        }

    market_data = reset_market_data(dataset)
    
    # Rolling Data Buffers
    window_grid_imp = deque(maxlen=50)
    window_demand = deque(maxlen=50)
    window_profit = deque(maxlen=50)
    
    total_grid_energy = 0.0
    total_energy_demand = 0.0
    step_count = 0
    CONFIG_FILE = "simulation_config.json"
    
    while True:
        # 1. Hot-Reload Check (Total Isolation)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                if config.get("dataset") != dataset:
                    dataset = config.get("dataset")
                    print(f"\n--- DATASET SWAP DETECTED: {dataset.upper()} ---")
                    env, obs = init_simulation(dataset)
                    market_data = reset_market_data(dataset)
                    window_grid_imp.clear()
                    window_demand.clear()
                    window_profit.clear()
                    total_grid_energy = 0.0
                    total_energy_demand = 0.0
                    step_count = 0
            except:
                pass

        # 2. Local Inference
        action, _ = model.predict(obs, deterministic=True)

        # 3. Environment Step
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # 4. Simulation Time Tracking
        sim_hour = int(env.current_idx % 24)
        market_data['sim_hour'] = sim_hour
        
        # 6. Update Analytics
        p2p_v = info.get('p2p_volume_kwh_step', 0)
        market_profit = info.get('market_profit_usd', 0)
        economic_profit = info.get('economic_profit_usd', 0)
        grid_imp = info.get('grid_import_kwh', 0)
        demand = info.get('total_demand_kw', 1e-4)
        
        intensity = 0.4
        co2_step = grid_imp * intensity
        
        market_data['total_p2p_volume'] += p2p_v
        market_data['cumulative_market_profit'] += market_profit
        market_data['cumulative_economic_profit'] += economic_profit
        market_data['cumulative_co2'] += co2_step
        market_data['cumulative_baseline_co2'] += (demand * intensity) 
        
        total_grid_energy += grid_imp
        total_energy_demand += demand
        market_data['cumulative_grid_dependency'] = (total_grid_energy / (total_energy_demand + 1e-9)) * 100.0
        
        window_grid_imp.append(grid_imp)
        window_demand.append(demand)
        window_profit.append(market_profit)
        
        market_data['rolling_grid_dependency'] = info.get('grid_dependency', 1.0) * 100.0
        market_data['rolling_market_profit'] = sum(window_profit)
        market_data['trade_success_rate'] = env.market_history['last_success_rate']
        market_data['last_update'] = datetime.now().strftime("%H:%M:%S")
        
        # Snapshot for Charts
        market_data['history'].append({
            'step': step_count,
            'p2p_v': p2p_v,
            'profit': market_profit,
            'grid_dep': market_data['rolling_grid_dependency'],
            'hour': sim_hour
        })
        if len(market_data['history']) > 50:
            market_data['history'].pop(0)

        # 7. Broadcast for Dashboard (Atomic write simulation)
        tmp_file = STATS_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(market_data, f, indent=2)
        os.replace(tmp_file, STATS_FILE)

        print(f"[{market_data['last_update']}] Step {step_count} (Hour {sim_hour:02d}:00)")
        
        if done or truncated:
            obs, _ = env.reset()
            
        time.sleep(1.5) # Optimized for high-level readability

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hybrid", choices=["hybrid", "ausgrid"])
    args = parser.parse_args()
    run_real_time(dataset=args.dataset)
