
import pandas as pd
import numpy as np
import argparse
import os

def merge_datasets(output_file="merged_training_data.csv", n_agents=4, days=30):
    """
    Creates a merged dataset by checking for available real data and falling back to synthetic.
    For Phase 2, we want to demonstrate we can handle the 'merging' concept.
    
    Since we only have one 'real' file (ausgrid_p2p_energy_dataset.csv) and one 'synthetic' (test_day_profile.csv),
    we will simulate a merge by:
    1. Loading the Ausgrid data.
    2. Duplicating/Shifting it to simulate more days/agents if needed.
    3. Filling gaps with synthetic generation.
    """
    
    print(f"Generating merged dataset: {output_file}...")
    
    # 1. Try to load Real Data
    real_data_path = "evaluation/ausgrid_p2p_energy_dataset.csv"
    if os.path.exists(real_data_path):
        print(f"Found real data at {real_data_path}")
        df_real = pd.read_csv(real_data_path)
    else:
        print("Real data not found. Creating placeholder.")
        df_real = pd.DataFrame()

    # 2. Generate/Augment
    # We want a robust training set.
    # Let's create a 30-day dataset.
    timestamps = pd.date_range(start="2024-01-01", periods=24*days, freq="h")
    
    merged_data = {
        "timestamp": timestamps
    }
    
    # Generate data for N agents
    for i in range(n_agents):
        # If we have real data, sample from it
        if not df_real.empty:
            # Pick a random column or reuse
            # Agent data usually: agent_0_demand_kw
            col_dem = f"agent_{i%4}_demand_kw" # Wrap around 4
            col_pv = f"agent_{i%4}_pv_kw"
            
            if col_dem in df_real.columns:
                # Tile the data to fill length
                vals_dem = np.resize(df_real[col_dem].values, len(timestamps))
                vals_pv = np.resize(df_real[col_pv].values, len(timestamps))
                
                # Add some noise to make it "different" agents if i > 4
                if i >= 4:
                    vals_dem *= np.random.uniform(0.9, 1.1, size=len(vals_dem))
                    vals_pv *= np.random.uniform(0.9, 1.1, size=len(vals_pv))
            else:
                # Fallback
                vals_dem = np.random.uniform(0.5, 2.0, size=len(timestamps))
                vals_pv = np.zeros(len(timestamps))
        else:
            # Synthetic Generation
            # Sinusoidal Pattern
            x = np.linspace(0, days*2*np.pi, len(timestamps))
            vals_pv = np.maximum(0, 5 * np.sin(x - np.pi) + np.random.normal(0, 0.5, len(timestamps))) # Day peak
            vals_dem = np.maximum(0, 2 + 1 * np.sin(x) + np.random.normal(0, 0.2, len(timestamps)))
            
        merged_data[f"agent_{i}_demand_kw"] = vals_dem
        merged_data[f"agent_{i}_pv_kw"] = vals_pv
        
    df_out = pd.DataFrame(merged_data)
    
    # Save
    df_out.to_csv(output_file, index=False)
    print(f"Saved {len(df_out)} rows to {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="merged_dataset_phase2.csv")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    merge_datasets(args.output, args.agents, args.days)
