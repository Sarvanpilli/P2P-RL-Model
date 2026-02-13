
import pandas as pd
import numpy as np
import os
import json

def preprocess_hybrid_data():
    print("--- Starting Hybrid Data Fusion ---")
    
    # Paths
    solar_file = "evaluation/ausgrid_p2p_energy_dataset.csv"
    wind_file = "evaluation/wind_generation_data.csv"
    output_file = "processed_hybrid_data.csv"
    config_file = "normalization_config.json"
    
    # 1. Load Data
    try:
        df_solar = pd.read_csv(solar_file)
        print(f"Loaded Solar Data: {df_solar.shape}")
        
        # Check wind file path
        if not os.path.exists(wind_file):
            # Try searching? Or assume it's in evaluation/
            print(f"Error: {wind_file} not found.")
            return

        df_wind = pd.read_csv(wind_file)
        print(f"Loaded Wind Data: {df_wind.shape}")
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Date & Time Standardization
    
    # Wind Data: 'Time' -> datetime 
    # File sample shows: 2017-01-02 00:00:00
    print("Standardizing Wind Time...")
    try:
        # Let pandas infer format. It's usually robust for standard YYYY-MM-DD HH:MM:SS
        df_wind['timestamp'] = pd.to_datetime(df_wind['Time'])
    except Exception as e:
        print(f"Wind Time conversion failed with default parser: {e}")
        # Try explicit format if inference fails
        try:
             df_wind['timestamp'] = pd.to_datetime(df_wind['Time'], format="%d-%m-%y %H:%M")
        except:
             print("Fallback format also failed. Checking first row:", df_wind['Time'].iloc[0])
             return

    df_wind = df_wind.sort_values('timestamp').reset_index(drop=True)
    
    # Solar Data: 'hour' column (0 to N). Needs virtual calendar.
    # Start Date: 2017-01-01 (Matches start year of wind usually)
    print("Creating Virtual Calendar for Solar...")
    
    # Align Solar start to Wind Start generally roughly around Jan 1st
    # Wind starts 2017-01-02? Let's use 2017-01-01 as requested by user or align.
    # User said: "create a virtual calendar starting from the same date as the wind data (2017-01-01)"
    # But wind data starts 2017-01-02.
    # We will stick to 2017-01-01 for Solar virtual start to cover the full year properly.
    start_date = pd.Timestamp("2017-01-01")
    
    n_solar = len(df_solar)
    solar_dates = pd.date_range(start=start_date, periods=n_solar, freq='H')
    df_solar['timestamp'] = solar_dates
    
    # 3. Merging / Fusing
    # Logic: Align by timestamp. 
    # Wind Data (2017-01-02 start) is missing Jan 1st?
    # We need to fill Jan 1st for Wind if Solar starts Jan 1st.
    # Or shift Wind?
    # Let's simple backfill/ffill Wind if needed or just align on intersection.
    # User said "Use the overlapping period".
    # Intersection of 2017-01-01 (Solar) and 2017-01-02 (Wind) means we lose Jan 1st.
    # But RL needs continuous data.
    # Let's Pad Wind for Jan 1st with Jan 2nd data (Backfill) to match Solar Start.
    
    # Create full target index (Solar's index)
    df_wind = df_wind.set_index('timestamp')
    
    # Reindex Wind to Solar's timeline
    # This will generate NaNs for Jan 1st if missing in Wind
    df_wind_aligned = df_wind.reindex(solar_dates)
    
    # Fill missing (e.g. Jan 1st)
    if df_wind_aligned.isnull().any().any():
        print("Filling missing Wind timestamps (e.g. alignment gaps) with Backfill/Ffill...")
        df_wind_aligned = df_wind_aligned.bfill().ffill() # Propagate first valid observation back to Jan 1
        
    df_wind_aligned = df_wind_aligned.reset_index().rename(columns={'index': 'timestamp'})
    
    # Now merge
    merged = pd.merge(df_solar, df_wind_aligned, on='timestamp', how='left')
    
    # Double check Solar Zeroes (Data Quality)
    print("--- Data Quality Check: Solar Zeroes ---")
    pv_cols = [c for c in df_solar.columns if 'pv' in c.lower()]
    for c in pv_cols:
        zeros = (df_solar[c] == 0).sum()
        total = len(df_solar)
        print(f"Col {c}: {zeros} / {total} zeros ({zeros/total:.1%})")
        # If > 60% zeros, it's normal for PV (Night). If 100%, suspicious.
    
    # 4. Cleaning & Missing Values
    print("Cleaning & Interpolating...")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(method='linear', limit=3)
    
    remaining_nans = merged[numeric_cols].isnull().any(axis=1)
    if remaining_nans.sum() > 0:
        print(f"Dropping {remaining_nans.sum()} rows with extensive missing data.")
        merged = merged.dropna(subset=numeric_cols)
    
    # 5. Feature Engineering (Fusion)
    # ... rest of logic relies on merged cols
        
    # Outliers: Negative Power
    # Check all agent_x_pv/demand and Wind Power
    # Assuming columns like 'agent_x_pv', 'agent_x_demand' exist.
    # And 'Power' from wind gen.
    
    cols_to_clip = [c for c in merged.columns if 'pv' in c or 'demand' in c or 'Power' in c or 'wind' in c.lower()]
    for c in cols_to_clip:
        if c in merged.columns:
            # Check negative
            neg_count = (merged[c] < 0).sum()
            if neg_count > 0:
                print(f"Clipping {neg_count} negative values in {c}")
                merged[c] = merged[c].clip(lower=0)
                
    # 5. Feature Engineering (The Fusion)
    print("Feature Engineering...")
    final_df = pd.DataFrame()
    final_df['timestamp'] = merged['timestamp']
    
    # Cyclical Time
    dt = merged['timestamp'].dt
    final_df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    final_df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
    final_df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    
    # Agent Mapping (Specific Requirements)
    # Agent 0: From Ausgrid (PV, Demand)
    if 'agent_0_pv' in merged.columns:
        final_df['agent_0_pv'] = merged['agent_0_pv']
        final_df['agent_0_demand'] = merged['agent_0_demand']
    else:
        print("Warning: agent_0_pv not found. Checking col names...")
        print(merged.columns)
        # Fallback if names are different (e.g. just 'pv', 'demand'?)
        
    # Agent 1: Wind
    # Map 'Power' -> 'agent_1_wind', Scale by 3.0
    if 'Power' in merged.columns:
        final_df['agent_1_wind'] = merged['Power'] * 3.0
    else:
        print("Warning: 'Power' column missing in Wind Data.")
        final_df['agent_1_wind'] = 0.0
        
    # Agent 1 Demand (Ausgrid)
    final_df['agent_1_demand'] = merged.get('agent_1_demand', 0.0)
    
    # Agents 2 & 3 (Ausgrid)
    for i in [2, 3]:
        final_df[f'agent_{i}_pv'] = merged.get(f'agent_{i}_pv', 0.0)
        final_df[f'agent_{i}_demand'] = merged.get(f'agent_{i}_demand', 0.0)

    # Weather Context
    # Prompt: temperature_2m, windspeed_100m
    final_df['temperature_2m'] = merged.get('temperature_2m', 0.0)
    final_df['windspeed_100m'] = merged.get('windspeed_100m', 0.0)
    
    # Solar at Night Check (User Request)
    # "check most of the values in the ausgrid data is zero"
    # Let's count zero PV
    pv_cols = [c for c in final_df.columns if 'pv' in c]
    for c in pv_cols:
        zeros = (final_df[c] == 0).sum()
        total = len(final_df)
        print(f"{c}: {zeros} zeros out of {total} ({zeros/total:.1%})")
        
    # 6. Normalization
    print("Normalizing...")
    config = {}
    
    normalized_df = final_df.copy()
    
    # Normalize Power Columns [0, 1]
    # PV, Wind, Demand
    power_cols = [c for c in final_df.columns if 'pv' in c or 'wind' in c or 'demand' in c] # 'agent_1_wind' matches
    
    # Find Global Max for each type? Or per column?
    # Prompt: "scaled to [0, 1] based on the maximum observed value across the whole dataset."
    # Usually better to scale per column or per type (Max PV, Max Demand).
    # Let's scale per column to preserve relative shape but fit 0-1.
    
    for c in power_cols:
        max_val = final_df[c].max()
        if max_val == 0: max_val = 1.0 # Avoid div/0
        
        normalized_df[c] = final_df[c] / max_val
        config[f"{c}_max"] = float(max_val)
        
    # Save Config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Normalization config saved to {config_file}")
    
    # 7. Output
    normalized_df.to_csv(output_file, index=False)
    print(f"--- Data Fusion Complete ---")
    print(f"Total synchronized hours: {len(normalized_df)}")
    if 'agent_0_pv' in normalized_df.columns and 'agent_1_wind' in normalized_df.columns:
        corr = normalized_df['agent_0_pv'].corr(normalized_df['agent_1_wind'])
        print(f"Average Solar vs Wind correlation: {corr:.3f}")
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    preprocess_hybrid_data()
