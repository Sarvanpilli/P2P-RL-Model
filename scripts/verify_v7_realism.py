
import os
import sys
import numpy as np
import pandas as pd

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def verify():
    print(">>> Verifying SLIM v7 Market Realism & Coordination")
    n = 4
    env = EnergyMarketEnvRobust(n_prosumers=2, n_consumers=2, data_file="processed_hybrid_data.csv")
    obs, _ = env.reset(seed=42)
    
    found_valid = False
    for t in range(168): # Look through a whole week if needed
        actions = []
        for i in range(n):
            trade = 1.0 if i < 2 else -1.0
            price_act = 0.0 if i < 2 else 1.0
            actions.append([0.0, trade, price_act])
        
        obs, reward, done, truncated, info = env.step(np.array(actions).flatten())
        
        # We need a step where there is both Demand - PV > 0 AND a P2P match
        # Let's check Baseline Import from info if it was there (v5 had it, v7 might not have it in info)
        # But we can check if total_import is less than what it would be? 
        # Actually I didn't expose total_baseline in info. I'll do that now.
        
        if info.get('p2p_volume', 0) > 0.1:
            # We found a match. Let's see if there was any baseline demand.
            # If Grid Reduction % > 0, we found it!
            if info.get('grid_reduction_percent', 0) > 0.01:
                print(f"\n--- Valid Efficiency Step Found at Step {t} ---")
                print(f"Price={info['market_price']:.4f}")
                print(f"P2P Volume: {info['p2p_volume']:.4f}")
                print(f"Total Import: {info['total_import']:.4f}")
                print(f"Grid Reduction %: {info['grid_reduction_percent']:.2%}")
                print(f"Global Grid Reward (Mean): {info.get('mean_global_grid_reward', 0):.6f}")
                print(f"Local Grid Reward (Mean): {info.get('mean_local_grid_reward', 0):.6f}")
                found_valid = True
                break
            
    if not found_valid:
        print("\nDEBUG: No high-efficiency steps found in the first 168 hours with current logic.")
        print("This might be because the agents aren't matched during peak demand hours.")
    else:
        print("\nVERIFICATION SUCCESSFUL: SLIM v7 efficiency signals are active.")

if __name__ == "__main__":
    verify()
