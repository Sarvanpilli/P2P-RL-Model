
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_phase3():
    print("Generating Phase 3 Plots...")
    
    # Load Data
    data = {}
    try:
        if pd.io.common.file_exists("results_phase2.csv"):
            data["Phase 2 (Legacy)"] = pd.read_csv("results_phase2.csv")
        if pd.io.common.file_exists("results_phase3.csv"):
            data["Phase 3 (Grid-Aware)"] = pd.read_csv("results_phase3.csv")
        if pd.io.common.file_exists("results_baseline.csv"):
            data["Baseline (Rule-Based)"] = pd.read_csv("results_baseline.csv")
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    if not data:
        print("No results found.")
        return

    # Combine
    combined = []
    for name, df in data.items():
        df["Agent"] = name
        combined.append(df)
    
    df_all = pd.concat(combined, ignore_index=True)
    
    # Set Style
    sns.set_theme(style="whitegrid")
    
    # 1. Grid Import Flow (Power)
    # We want to show explicitly IMPORT Power.
    # Metric: Step 'total_import' (kW)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_all, x="step", y="total_import", hue="Agent")
    plt.title("Grid Import Minimization: Phase 3 vs Others")
    plt.ylabel("Grid Import (kW)")
    plt.xlabel("Step (Hour)")
    plt.tight_layout()
    plt.savefig("phase3_grid_import.png")
    
    # 2. Cumulative Reward
    # Add cumsum
    for df in combined:
        df["cum_reward"] = df["total_reward"].cumsum()
    df_cum = pd.concat(combined, ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_cum, x="step", y="cum_reward", hue="Agent")
    plt.title("Cumulative Reward: Grid-Aware Optimization")
    plt.ylabel("Cumulative Reward (Profit - Penaltes)")
    plt.xlabel("Step (Hour)")
    plt.tight_layout()
    plt.savefig("phase3_cumulative_reward.png")
    
    # 3. Grid Independence Ratio (Bar Chart)
    # 1 - (Total Import / Total Demand)
    # Total Demand? We don't have it directly in results CSV.
    # But info["total_import"] is logged.
    # Let's assume baseline import is "Demand - PV" roughly?
    # Phase 3 should have LOWER Total Import than Baseline.
    # Let's just plot Total Import Sum.
    
    import_sums = df_all.groupby("Agent")["total_import"].sum().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=import_sums, x="Agent", y="total_import", palette="viridis")
    plt.title("Total Grid Import (Lower is Better)")
    plt.ylabel("Total Import (kWh)")
    plt.tight_layout()
    plt.savefig("phase3_total_import.png")
    
    print("Saved plots: phase3_grid_import.png, phase3_cumulative_reward.png, phase3_total_import.png")

if __name__ == "__main__":
    plot_phase3()
