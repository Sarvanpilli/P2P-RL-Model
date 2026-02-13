
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison():
    print("Generating Comparison Plots...")
    
    # Load Data
    try:
        df_rl = pd.read_csv("results_rl.csv")
        df_base = pd.read_csv("results_baseline.csv")
    except FileNotFoundError:
        print("Error: Could not find results CSVs.")
        return

    # Add labels
    df_rl["Agent"] = "RL (PPO)"
    df_base["Agent"] = "Baseline (Rule-Based)"
    
    # Combine
    df = pd.concat([df_rl, df_base], ignore_index=True)
    
    # Set Style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Market Price Stability
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="market_price", hue="Agent", style="Agent", markers=True)
    plt.title("Market Price Stability: RL vs Baseline")
    plt.ylabel("Price ($/kWh)")
    plt.xlabel("Step (Hour)")
    plt.tight_layout()
    plt.savefig("comparison_market_price.png")
    print("Saved comparison_market_price.png")
    
    # Plot 2: Distribution Losses (Efficiency)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="loss_kw", hue="Agent", style="Agent", markers=True)
    plt.title("Grid Distribution Losses: RL vs Baseline")
    plt.ylabel("Losses (kW)")
    plt.xlabel("Step (Hour)")
    plt.tight_layout()
    plt.savefig("comparison_losses.png")
    print("Saved comparison_losses.png")
    
    # Plot 3: Cumulative Reward (Profit/Utility)
    # create cumulative column
    df_rl["cum_reward"] = df_rl["total_reward"].cumsum()
    df_base["cum_reward"] = df_base["total_reward"].cumsum()
    
    df_cum = pd.concat([df_rl, df_base], ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cum, x="step", y="cum_reward", hue="Agent", style="Agent", markers=True)
    plt.title("Cumulative Social Welfare (Reward): RL vs Baseline")
    plt.ylabel("Cumulative Reward")
    plt.xlabel("Step (Hour)")
    plt.tight_layout()
    plt.savefig("comparison_cumulative_reward.png")
    print("Saved comparison_cumulative_reward.png")

if __name__ == "__main__":
    plot_comparison()
