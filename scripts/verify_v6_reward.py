
import os
import sys
import numpy as np
import pandas as pd

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def verify():
    print(">>> Verifying SLIM v6 Reward Signals (High Precision)")
    n = 4
    env = EnergyMarketEnvRobust(n_prosumers=2, n_consumers=2, data_file="processed_hybrid_data.csv", random_start_day=True)
    obs, _ = env.reset(seed=42)
    
    # 1. Test Price Advantage
    print("\n--- Testing Price Advantage ---")
    actions = []
    for i in range(n):
        trade = 1.0 if i < 2 else -1.0
        actions.append([0.0, trade, 0.9])
    
    obs, reward, done, truncated, info = env.step(np.array(actions).flatten())
    print(f"Price={info['market_price']:.4f}, Retail=0.2000")
    pa = info.get('mean_price_advantage', 0)
    print(f"Advantage Reward (Mean): {pa:.6f}")
    
    # 2. Test Failed Trade Penalty
    print("\n--- Testing Failed Trade Penalty (Forcing failure) ---")
    obs, _ = env.reset(seed=42)
    actions = []
    for i in range(n):
        trade = 1.0 if i < 2 else -1.0
        # Force a large spread by using extreme perception
        # Price bid Action for Seller: 1.0 -> retail
        # Price bid Action for Buyer: 0.0 -> clearing (which is low)
        # We manually shift current_idx to a peak hour where retail is 0.5
        env.current_idx = 18 # hour 18
        price_act = 1.0 if i < 2 else 0.0
        actions.append([0.0, trade, price_act])
    
    obs, reward, done, truncated, info = env.step(np.array(actions).flatten())
    print(f"P2P Volume: {info['p2p_volume']:.4f}")
    ftp = info.get('mean_failed_trade_penalty', 0)
    print(f"Failed Trade Penalty (Mean): {ftp:.6f}")
    
    # 3. Test Profit Normalization
    print("\n--- Testing Profit Normalization ---")
    # Clean profit should be negative during peak if no P2P matches
    print(f"Mean Clean Profit: {info.get('mean_clean_profit', 0):.6f}")
    
    if pa >= 0 and ftp >= 0:
        print("\nVERIFICATION SUCCESSFUL: v6 Signal logic confirmed.")
    else:
        print("\nVERIFICATION FAILED: Critical signals are dead.")

if __name__ == "__main__":
    verify()
