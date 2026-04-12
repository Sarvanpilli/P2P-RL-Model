
import os
import sys
import time
import json
import requests
import numpy as np
from datetime import datetime

# Custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train.energy_env_robust import EnergyMarketEnvRobust

API_URL = "http://127.0.0.1:8000/predict"
STATS_FILE = "live_market_stats.json"

def run_real_time():
    print("Starting Real-Time Simulation Loop (Streaming Data)...")
    
    # Init Env
    env = EnergyMarketEnvRobust(
        n_agents=4, 
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
    )
    
    # Load Baseline to compare reduction
    baseline_stats = {}
    if os.path.exists("research_q1/results/carbon_baseline.json"):
        with open("research_q1/results/carbon_baseline.json", "r") as f:
            baseline_stats = json.load(f)
    baseline_co2_avg = baseline_stats.get('avg_co2_per_step', 0.1)

    obs, _ = env.reset()
    
    # Persistent State for Dashboard
    market_data = {
        'total_p2p_volume': 0.0,
        'cumulative_market_profit': 0.0,
        'cumulative_economic_profit': 0.0,
        'rolling_market_profit': 0.0,
        'cumulative_co2': 0.0,
        'cumulative_baseline_co2': 0.0,
        'rolling_grid_dependency': 100.0,
        'cumulative_grid_dependency': 100.0,
        'trade_success_rate': 0.0,
        'is_active': True,
        'last_update': "",
        'history': []
    }
    
    # Deques for rolling windows
    from collections import deque
    window_grid_imp = deque(maxlen=50)
    window_demand = deque(maxlen=50)
    window_profit = deque(maxlen=50)
    
    total_grid_energy = 0.0
    total_energy_demand = 0.0

    step_count = 0
    while True:
        try:
            # 1. Get Action from API
            resp = requests.post(API_URL, json={"observation": obs.tolist()}, timeout=1)
            if resp.status_code == 200:
                action = np.array(resp.json()['action'])
            else:
                print(f"API Error {resp.status_code}. Using random fallback.")
                action = env.action_space.sample()
        except requests.exceptions.ConnectionError:
            print("API Offline. Retrying in 2s...")
            time.sleep(2)
            continue

        # 2. Step Environment
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # 3. Update Stats (Scientific Corrections)
        p2p_v = info.get('p2p_volume_kwh_step', 0)
        market_profit = info.get('market_profit_usd', 0)
        economic_profit = info.get('economic_profit_usd', 0)
        grid_imp = info.get('grid_import_kwh', 0)
        demand = info.get('total_demand_kw', 1e-4)
        intensity = info.get('carbon_intensity_kg_kwh', 0.5)
        
        # Physical Consistency Checks
        if float(grid_imp) > 0 and p2p_v == 0 and market_profit == 0:
            print(f"CRITICAL WARNING: Grid import detected ({grid_imp:.2f}) but P2P/Profit is zero. Model might be in standby.")

        co2_step = grid_imp * intensity
        
        market_data['total_p2p_volume'] += p2p_v
        market_data['cumulative_market_profit'] += market_profit
        market_data['cumulative_economic_profit'] += economic_profit
        market_data['cumulative_co2'] += co2_step
        market_data['cumulative_baseline_co2'] += baseline_co2_avg
        
        # Grid Dependency Calculations (Corrected formula)
        total_grid_energy += grid_imp
        total_energy_demand += demand
        market_data['cumulative_grid_dependency'] = (total_grid_energy / (total_energy_demand + 1e-9)) * 100.0
        
        window_grid_imp.append(grid_imp)
        window_demand.append(demand)
        window_profit.append(market_profit)
        
        market_data['rolling_grid_dependency'] = (sum(window_grid_imp) / (sum(window_demand) + 1e-9)) * 100.0
        market_data['rolling_market_profit'] = sum(window_profit)
        
        market_data['trade_success_rate'] = env.market_history['last_success_rate']
        market_data['last_update'] = datetime.now().strftime("%H:%M:%S")
        
        # Snapshot for charts (limit to last 50)
        market_data['history'].append({
            'step': step_count,
            'p2p_v': p2p_v,
            'profit': market_profit,
            'co2': co2_step,
            'grid_dep': market_data['rolling_grid_dependency']
        })
        if len(market_data['history']) > 50:
            market_data['history'].pop(0)

        # 4. Save for Dashboard
        with open(STATS_FILE, "w") as f:
            json.dump(market_data, f, indent=2)

        print(f"[{market_data['last_update']}] Step {step_count} | Vol: {p2p_v:.2f} | Mkt_Profit: {market_profit:.2f} | Success: {info.get('match_info',{}).get('total_volume',0)>0}")
        
        if done or truncated:
            print("Resetting Episode...")
            obs, _ = env.reset()
            
        time.sleep(1) # Real-time simulation delay

if __name__ == "__main__":
    run_real_time()
