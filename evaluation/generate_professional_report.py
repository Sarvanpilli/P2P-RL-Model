import os
import pandas as pd
import numpy as np

def get_tou_prices(hour):
    """Returns (retail, feed_in) for a given hour."""
    # From energy_env_robust.py
    if 0 <= hour < 6 or hour >= 23:
        return 0.10, 0.03 # Off-Peak
    elif 17 <= hour < 21:
        return 0.35, 0.15 # Peak
    else:
        return 0.20, 0.08 # Standard

def calculate_baseline_metrics(data_path):
    """Calculates metrics for a 'No Battery, Grid-Only' baseline."""
    df = pd.read_csv(data_path)
    
    total_cost = 0.0
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    
    # Assume 4 agents as per dataset columns
    agents = [0, 1, 2, 3] # Adjusted based on known dataset structure
    
    print("Calculating Baseline (No Battery, No P2P)...")
    
    for idx, row in df.iterrows():
        hour = int(row['hour']) % 24
        retail, feed_in = get_tou_prices(hour)
        
        step_import = 0.0
        step_export = 0.0
        step_cost = 0.0
        
        for i in agents:
            # Safe access
            dem = row.get(f'agent_{i}_demand_kw', 0.0)
            pv = row.get(f'agent_{i}_pv_kw', 0.0)
            
            net = dem - pv
            
            if net > 0:
                # Must buy from grid
                cost = net * retail
                step_import += net
                step_cost += cost
            else:
                # Excess to grid
                revenue = abs(net) * feed_in
                step_export += abs(net)
                step_cost -= revenue
        
        total_cost += step_cost
        total_import_kwh += step_import
        total_export_kwh += step_export
        
    return {
        "cost": total_cost,
        "import": total_import_kwh,
        "export": total_export_kwh
    }

def analyze_rl_results(results_path):
    df = pd.read_csv(results_path)
    
    # RL Cost Estimation
    # We need to reconstruct cost from Grid Flow because 'total_reward' mixes many things.
    # We don't have per-step Grid Cost logged explicitly in CSV? 
    # We have 'total_import' and 'total_export' (Community Aggregate).
    # We can apply ToU prices to these aggregates to get Community Cashflow.
    
    total_cost = 0.0
    
    for idx, row in df.iterrows():
        # Step is index roughly?
        step = row['step']
        hour = int(step) % 24
        retail, feed_in = get_tou_prices(hour)
        
        imp = row['total_import']
        exp = row['total_export']
        
        # Cost = Import * Retail - Export * FeedIn
        step_bill = (imp * retail) - (exp * feed_in)
        total_cost += step_bill
        
    return {
        "cost": total_cost,
        "import": df['total_import'].sum(),
        "export": df['total_export'].sum(),
        "avg_price": df['market_price'].mean(),
        "steps": len(df)
    }

if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(_dir, "ausgrid_p2p_energy_dataset.csv")
    RESULTS_PATH = os.path.join(_dir, "evaluation_results.csv")
    
    baseline = calculate_baseline_metrics(DATA_PATH)
    rl = analyze_rl_results(RESULTS_PATH)
    
    # Report
    with open("final_report.txt", "w") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("CRITICAL EVALUATION REPORT\n")
        f.write("="*50 + "\n")
        
        f.write(f"\n1. PHYSICAL UNITS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Simulation Duration: {rl['steps']} Hours (~{rl['steps']/24:.1f} Days)\n")
        f.write("All 'Power' metrics sum to 'Energy' (kWh) due to 1-hour timesteps.\n")
        
        f.write(f"\n2. FINANCIAL COMPARISON (Community Level)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<20} | {'Baseline (No Batt)':<18} | {'P2P-RL System':<18} | {'Improvement':<12}\n")
        f.write("-" * 75 + "\n")
        
        # Cost
        base_cost = baseline['cost']
        rl_cost = rl['cost']
        cost_saving = base_cost - rl_cost
        cost_imp_pct = (cost_saving / abs(base_cost)) * 100 if base_cost != 0 else 0
        
        f.write(f"{'Net Bill ($)':<20} | {base_cost:18.2f} | {rl_cost:18.2f} | {cost_imp_pct:11.1f}%\n")
        
        # Energy
        f.write(f"{'Grid Import (MWh)':<20} | {baseline['import']/1000:18.3f} | {rl['import']/1000:18.3f} | {-(baseline['import']-rl['import'])/baseline['import']*100:11.1f}%\n")
        f.write(f"{'Grid Export (MWh)':<20} | {baseline['export']/1000:18.3f} | {rl['export']/1000:18.3f} | {(rl['export']-baseline['export'])/baseline['export']*100:11.1f}%\n")
        
        f.write("\n3. MARKET PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Clearing Price: ${rl['avg_price']:.3f}/kWh\n")
        # Compare to avg grid price
        avg_retail = (0.10 * 7 + 0.35 * 4 + 0.20 * 13) / 24
        avg_feed = (0.03 * 7 + 0.15 * 4 + 0.08 * 13) / 24
        f.write(f"vs Grid Retail Avg:     ${avg_retail:.3f}/kWh\n")
        f.write(f"vs Grid Feed-in Avg:    ${avg_feed:.3f}/kWh\n")
        
        f.write(f"\n4. VERDICT\n")
        f.write("-" * 20 + "\n")
        if rl_cost < base_cost:
            f.write("SUCCESS: The P2P-RL system outperforms the baseline.\n")
        else:
            f.write("FAILURE: The P2P-RL system is more expensive than baseline.\n")
            
        f.write("\nFor comparison:\n")
        f.write(f"Baseline Net Bill: ${base_cost:.2f}\n")
        f.write(f"RL Net Bill:       ${rl_cost:.2f}\n")
