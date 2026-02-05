import pandas as pd
import numpy as np

df = pd.read_csv("evaluation/evaluation_results.csv")


with open("results.txt", "w") as f:
    f.write("--- Evaluation Feedback ---\n")
    f.write(f"Total Steps: {len(df)}\n")
    f.write(f"Total Reward: {df['total_reward'].sum():.2f}\n")
    f.write(f"Avg Market Price: ${df['market_price'].mean():.4f}/kWh\n")
    f.write(f"Total Grid Import: {df['total_import'].sum():.2f} kW\n")
    f.write(f"Total Grid Export: {df['total_export'].sum():.2f} kW\n")

    # Calculate Net Grid Interation
    net_grid = df['total_import'].sum() - df['total_export'].sum()
    f.write(f"Net Grid Interaction: {net_grid:.2f} kW (+Import/-Export)\n")

    # Check Constraints (Soc)
    cols = [c for c in df.columns if 'soc' in c]
    for c in cols:
        min_soc = df[c].min()
        max_soc = df[c].max()
        f.write(f"{c}: Range [{min_soc:.2f}, {max_soc:.2f}]\n")


