
import pandas as pd
import numpy as np
import os
import argparse

def generate_report(csv_path="evaluation/evaluation_results.csv", output_md="evaluation/research_report.md"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # --- Compute Metrics ---
    hours = len(df)
    n_agents = 4 # Detect?
    
    # 1. Economic Indices
    if "total_reward" in df.columns:
        total_reward = df["total_reward"].sum()
        avg_reward_per_step = df["total_reward"].mean()
    else:
        total_reward = 0.0
        avg_reward_per_step = 0.0

    # 2. Grid Independence
    # Total Import vs Total Demand
    # We need to sum per-agent demands. Columns like "agent_0_demand"
    demand_cols = [c for c in df.columns if "_demand" in c]
    total_demand_energy = df[demand_cols].sum().sum()
    
    # Imports
    # Look for "grid_flow" or explicit import columns.
    # grid_flow = Export - Import.
    # But for Independence, we need Gross Import.
    # In evaluate_episode.py I logged 'agent_i_grid_trade'.
    trade_cols = [c for c in df.columns if "_grid_trade" in c]
    
    total_import_energy = 0.0
    total_export_energy = 0.0
    
    for c in trade_cols:
        col_data = df[c]
        imports = col_data[col_data < 0].sum() # Negative is import
        exports = col_data[col_data > 0].sum()
        total_import_energy += abs(imports)
        total_export_energy += exports
        
    independence_index = 1.0 - (total_import_energy / (total_demand_energy + 1e-6))
    independence_index = max(0.0, independence_index) # clip
    
    # 3. Fairness (Avg Gini)
    if "gini" in df.columns:
        avg_gini = df["gini"].mean()
    else:
        avg_gini = 0.0
        
    # 4. Carbon Footprint
    if "total_co2" in df.columns:
        total_co2 = df["total_co2"].sum()
    else:
        total_co2 = 0.0

    # 5. Overload Frequency
    # "grid_flow" vs limit (200kW).
    # But usually we log "overload_kw" or we can infer it.
    # Assuming limit is 200.0 from default.
    limit = 200.0
    # Actually, grid_flow is NET. We need Total Export/Import to check limit.
    # We don't have total_export column explicitly in standard csv unless we added it.
    # We can reconstruct from trade_cols.
    
    overload_events = 0
    for i in range(len(df)):
        step_export = 0.0
        step_import = 0.0
        for c in trade_cols:
            val = df.iloc[i][c]
            if val > 0: step_export += val
            else: step_import += abs(val)
        
        if step_export > limit or step_import > limit:
            overload_events += 1
            
    overload_freq = overload_events / len(df)

    # --- Generate Markdown ---
    
    md = f"""# P2P Energy Trading: Research Report

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Data Source**: `{csv_path}`

## 1. Executive Summary

| Metric | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| **Grid Independence** | {independence_index:.2%} | % | Portion of demand met by local/P2P resources. |
| **Average Gini** | {avg_gini:.3f} | [0,1] | Inequality in profit distribution (Lower is better). |
| **Total Carbon Emissions** | {total_co2:.2f} | kg | Total CO2 generated from grid imports. |
| **Grid Overload Freq** | {overload_freq:.2%} | % | Steps where line capacity was exceeded. |
| **Net Social Welfare** | {total_reward:.2f} | $ | Total cumulative reward (Proxy for economic efficiency). |

## 2. Detailed Verification

### A. Energy Conservation
*   **Total Demand**: {total_demand_energy:.2f} kWh
*   **Total Imported**: {total_import_energy:.2f} kWh
*   **Total Exported**: {total_export_energy:.2f} kWh
*   **Net Grid Interaction**: {total_export_energy - total_import_energy:.2f} kWh

### B. Agent Performance
"""
    
    # Per Agent Stats
    for i in range(len(demand_cols)):
        agent_p = f"agent_{i}_profit"
        agent_s = f"agent_{i}_soc"
        
        profit = df[agent_p].sum() if agent_p in df.columns else 0.0
        avg_soc = df[agent_s].mean() if agent_s in df.columns else 0.0
        
        md += f"*   **Agent {i}**: Profit = ${profit:.2f}, Avg SoC = {avg_soc:.1f} kWh\n"

    with open(output_md, "w") as f:
        f.write(md)
        
    print(f"Report generated: {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="evaluation/evaluation_results.csv")
    parser.add_argument("--out", default="evaluation/research_report.md")
    args = parser.parse_args()
    
    generate_report(args.csv, args.out)
