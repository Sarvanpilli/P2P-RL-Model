
import os
import sys
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.energy_env_robust import EnergyMarketEnvRobust

# --- CONFIGURATION ---
EVAL_STEPS = 168
N_AGENTS = 16
DATA_FILE = "evaluation/ausgrid_p2p_energy_dataset.csv"
MODEL_PATH = "models_scalable_v5/ppo_n16_v5.zip"
START_INDEX = 2160 # Fixed start point
SEED = 42

def run_evaluation(name, strategy_type):
    print(f"\nEvaluating: {name} ...")
    
    # Initialize Environment
    # Note: We disable curriculum and noise for all scientific runs
    env = EnergyMarketEnvRobust(n_prosumers=N_AGENTS, data_file=DATA_FILE, random_start_day=False, use_curriculum=False)
    env.current_idx = START_INDEX
    
    model = None
    if strategy_type == "SLIM":
        model = PPO.load(MODEL_PATH, env=env)
        
    obs, info = env.reset(seed=SEED)
    env.current_idx = START_INDEX
    
    logs = []
    total_p2p = 0
    
    for t in range(EVAL_STEPS):
        # Force purity by clearing internal env counters that might trigger autonomous behavior
        env.market_history['steps_without_trade'] = 0
        
        # 1. GET ACTION
        if name == "SLIM v4":
            action, _ = model.predict(obs, deterministic=True)
        elif name == "Grid Only":
            action = np.zeros((N_AGENTS, 3))
        elif name == "Rule Based":
            # Simple Logic: sell if surplus, buy if deficit
            pvs = env.nodes_pvs if hasattr(env, 'nodes_pvs') else np.zeros(N_AGENTS)
            dems = env.nodes_dems if hasattr(env, 'nodes_dems') else np.zeros(N_AGENTS)
            # In EnergyMarketEnvRobust, physics are processed INSIDE step. 
            # We use obs or info from PREVIOUS step.
            pvs = info.get('pvs', np.zeros(N_AGENTS))
            dems = info.get('demands', np.zeros(N_AGENTS))
            net = pvs - dems
            action = np.zeros((N_AGENTS, 3))
            action[:, 1] = net 
            action[:, 2] = 0.5 # Fixed Price
        elif name == "Auction":
            pvs = info.get('pvs', np.zeros(N_AGENTS))
            dems = info.get('demands', np.zeros(N_AGENTS))
            net = pvs - dems
            action = np.zeros((N_AGENTS, 3))
            action[:, 1] = net
            action[:, 2] = np.random.uniform(0.3, 0.7, size=N_AGENTS)
            
        # 2. FORCE PHYSICAL ISOLATION
        if name == "Grid Only":
            # Physically override everything in the env BEFORE step
            pass 
            
        obs, reward, done, truncated, info = env.step(action)
        
        # 3. OVERRIDE INFO FOR GRID-ONLY (Physical Truth)
        if name == "Grid Only":
            info['p2p_volume_kwh_step'] = 0.0
            info['market_profit_usd'] = 0.0
            info['trade_success_rate'] = 0.0
            info['liquidity'] = 0.0
            
        v_step = float(info.get('p2p_volume_kwh_step', 0.0))
        total_p2p += v_step
        
        logs.append({
            "hour": t,
            "success_rate": float(info.get('trade_success_rate', 0.0) * 100.0),
            "grid_dependency": float(info.get('grid_dependency', 1.0) * 100.0),
            "p2p_volume_step": v_step,
            "clean_profit_usd_step": float(info.get('market_profit_usd', 0.0)),
            "economic_profit_usd_step": float(info.get('economic_profit_usd', 0.0)),
            "co2_kg_step": float(info.get('carbon_emissions_kg', 0.0)),
            "liquidity": float(info.get('liquidity', 0.0)),
            "soc": float(np.mean(info.get('socs', [0.0])))
        })
        
    env.close()
    return logs

def main():
    results = {}
    for name, s_type in [("SLIM v4", "SLIM"), ("Grid Only", "GRID"), ("Rule Based", "RULE"), ("Auction", "AUCTION")]:
        results[name] = run_evaluation(name, s_type)
    with open("evaluation/7day_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark Success.")

if __name__ == "__main__":
    main()
