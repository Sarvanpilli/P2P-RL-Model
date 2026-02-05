import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=5, help="Total number of agents")
    args = parser.parse_args()
    
    N_AGENTS = args.n_agents
    print(f"Generating scenario data for {N_AGENTS} agents...")
    
    # Time vector (24 hours, 1 hour steps)
    t = np.arange(24)
    
    data = {}
    
    # Base Patterns
    # Solar peak at t=12 (Noon)
    # Demand peak at t=19 (Evening)
    solar_base_curve = np.maximum(0, np.sin(np.pi * (t - 6) / 12)) 
    demand_base_curve = np.maximum(0.2, 0.5 + 0.5 * np.sin(np.pi * (t - 14) / 12)) # Peak around 20:00
    
    rng = np.random.default_rng(42)
    
    for i in range(N_AGENTS):
        # 1. Solar Profile
        # 80% have solar, 20% might have zero (pure consumer via data essentially, 
        # though explicit type enforces it too)
        has_solar = rng.random() > 0.1 
        if has_solar:
             peak_kw = rng.uniform(4, 15)
             pv = peak_kw * solar_base_curve
             # Add weather noise
             noise = rng.normal(0, 0.5, 24)
             pv = np.maximum(0, pv + noise)
        else:
             pv = np.zeros(24)
             
        # 2. Demand Profile
        avg_load = rng.uniform(2, 8)
        dem = avg_load * demand_base_curve
        noise = rng.normal(0, 1.0, 24)
        dem = np.maximum(0.1, dem + noise)
        
        data[f"agent_{i}_pv_kw"] = pv
        data[f"agent_{i}_demand_kw"] = dem
        
    df = pd.DataFrame(data)
    
    # Save
    os.makedirs('scenarios', exist_ok=True)
    df.to_csv('scenarios/user_scenario_data.csv', index=False)
    print(f"Saved to scenarios/user_scenario_data.csv with {len(df.columns)//2} agents.")
