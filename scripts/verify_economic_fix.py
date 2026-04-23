
import os
import sys
import numpy as np
import pandas as pd

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def verify():
    print(">>> Verifying Economic Fix (v4) - Random Search for Trade Opportunity")
    n = 4
    # Use real data file
    env = EnergyMarketEnvRobust(n_prosumers=2, n_consumers=2, data_file="processed_hybrid_data.csv", random_start_day=True)
    
    # Try multiple episodes to find one with trades
    for episode in range(5):
        obs, _ = env.reset()
        print(f"\nEpisode {episode} Start (Retail={env._get_grid_prices()[0]})")
        
        has_trades_this_ep = False
        for t in range(24):
            actions = []
            for i in range(n):
                # Aggressive bidding to force matches if energy exists
                trade = 1.0 if i < 2 else -1.0 # Prosumers sell, Consumers buy
                actions.append([0.0, trade, 0.9])
            
            obs, reward, done, truncated, info = env.step(np.array(actions).flatten())
            
            p2p_v = info.get('p2p_volume', 0)
            if p2p_v > 1e-4:
                print(f"  Step {t}: P2P Vol={p2p_v:.2f}, Grid Redu={info.get('grid_reduction_percent', 0):.2f}, Clean Prof={info.get('mean_clean_profit', 0):.4f}")
                has_trades_this_ep = True
                
        if has_trades_this_ep:
            print("\nVERIFICATION SUCCESSFUL: P2P Trades occurring in this episode.")
            return

    print("\nVERIFICATION FAILED: No P2P volume found in 5 random episodes. Critical logic check needed.")

if __name__ == "__main__":
    verify()
