
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison():
    # Load Data
    files = {
        "Phase 3 (Benchmark)": "results_phase3_bench.csv",
        "Phase 4 (Predictive)": "results_phase4.csv"
    }
    
    dfs = []
    for mode, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Configuration'] = mode
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")

    if not dfs:
        print("No data to plot.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # 1. Cumulative Reward Comparison
    plt.figure(figsize=(10, 6))
    for mode in combined_df['Configuration'].unique():
        subset = combined_df[combined_df['Configuration'] == mode]
        plt.plot(subset['step'], subset['total_reward'].cumsum(), label=mode)
    
    plt.title("Cumulative Reward: Phase 3 vs Phase 4")
    plt.xlabel("Step (Hours)")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("phase4_cumulative_reward.png")
    plt.close()

    # 2. Smoothing Analysis (Jitter)
    # If we have "smoothing_penalty", let's plot it directly
    if 'smoothing_penalty' in combined_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=combined_df, x='step', y='smoothing_penalty', hue='Configuration')
        plt.title("Action Smoothing Penalty (Jitter)")
        plt.xlabel("Step (Hours)")
        plt.ylabel("Penalty Magnitude")
        plt.grid(True, alpha=0.3)
        plt.savefig("phase4_jitter_penalty.png")
        plt.close()
    
    # 3. Peak Hour Grid Import (17:00 - 21:00)
    # Filter for hours 17-21
    combined_df['hour'] = combined_df['step'] % 24
    peak_df = combined_df[(combined_df['hour'] >= 17) & (combined_df['hour'] < 21)]
    
    if not peak_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=peak_df, x='Configuration', y='total_import')
        plt.title("Grid Import Distribution during Peak Hours (17:00 - 21:00)")
        plt.ylabel("Total Import (kW)")
        plt.grid(True, alpha=0.3)
        plt.savefig("phase4_peak_import.png")
        plt.close()

    # 4. SoC Profile Comparison
    if 'soc_mean' in combined_df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_df, x='step', y='soc_mean', hue='Configuration')
        plt.title("Average Battery SoC Profile")
        plt.xlabel("Step (Hours)")
        plt.ylabel("Mean SoC (kWh)")
        plt.grid(True, alpha=0.3)
        plt.savefig("phase4_soc_profile.png")
        plt.close()

    print("Plots generated: phase4_cumulative_reward.png, phase4_jitter_penalty.png, phase4_peak_import.png, phase4_soc_profile.png")

if __name__ == "__main__":
    plot_comparison()
