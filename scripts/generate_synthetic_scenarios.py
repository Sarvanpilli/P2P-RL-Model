
import pandas as pd
import numpy as np
import os

def generate_scenarios():
    input_file = "processed_hybrid_data.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    os.makedirs("scenarios", exist_ok=True)

    # Helper to get hour from timestamp if available, else use index % 24
    if 'timestamp' in df.columns:
        df['hour_val'] = pd.to_datetime(df['timestamp']).dt.hour
    else:
        df['hour_val'] = df.index % 24

    # --- SCENARIO 1: GRID STRESS (DEFICIT) ---
    df_deficit = df.copy()
    
    # Identify agent columns
    demand_cols = [c for c in df.columns if 'demand' in c]
    gen_cols = [c for c in df.columns if 'pv' in c or 'wind' in c]

    for col in demand_cols:
        # demand = demand * Uniform(1.3, 1.6)
        noise = np.random.uniform(1.3, 1.6, size=len(df_deficit))
        df_deficit[col] = df_deficit[col] * noise
        
        # Apply peak-hour stress: [18, 19, 20, 21]
        mask = df_deficit['hour_val'].isin([18, 19, 20, 21])
        df_deficit.loc[mask, col] *= 1.3

    for col in gen_cols:
        # generation = generation * Uniform(0.3, 0.6)
        noise = np.random.uniform(0.3, 0.6, size=len(df_deficit))
        df_deficit[col] = df_deficit[col] * noise

    df_deficit.drop(columns=['hour_val'], inplace=True)
    df_deficit.to_csv("scenarios/scenario_deficit.csv", index=False)
    print("Saved scenarios/scenario_deficit.csv")

    # --- SCENARIO 2: HIGH P2P LIQUIDITY ---
    df_p2p = df.copy()
    
    # Implementation:
    # For each agent i:
    # if i % 2 == 0: generation_i *= 1.8, demand_i *= 0.8
    # else: generation_i *= 0.5, demand_i *= 1.5
    
    # Note: processed_hybrid_data typically has agents 0, 1, 2, 3
    for i in range(4): # Basic 4 agents in the CSV
        d_col = f"agent_{i}_demand"
        if d_col not in df_p2p.columns:
            d_col = f"agent_{i}_demand_kw" # Check alternate naming
            
        g_col = f"agent_{i}_pv"
        if g_col not in df_p2p.columns:
            g_col = f"agent_{i}_pv_kw"
        if g_col not in df_p2p.columns and i == 1:
            g_col = "agent_1_wind" # Special case for wind agent
            
        if d_col in df_p2p.columns and g_col in df_p2p.columns:
            if i % 2 == 0:
                df_p2p[g_col] *= 1.8
                df_p2p[d_col] *= 0.8
            else:
                df_p2p[g_col] *= 0.5
                df_p2p[d_col] *= 1.5
                
    # Also adjust temperature/wind speed slightly to match the "feel" (optional but good)
    # Wind speeds up in high P2P if agent 1 has wind? User didn't ask, so we skip.

    df_p2p.drop(columns=['hour_val'], inplace=True)
    df_p2p.to_csv("scenarios/scenario_high_p2p.csv", index=False)
    print("Saved scenarios/scenario_high_p2p.csv")

if __name__ == "__main__":
    generate_scenarios()
