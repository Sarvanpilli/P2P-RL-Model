
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_real():
    print("Generating Real Data Plots...")
    
    try:
        df_rl = pd.read_csv("real_results_rl.csv")
        df_base = pd.read_csv("real_results_baseline.csv")
    except FileNotFoundError:
        print("CSVs not found.")
        return

    df_rl["Agent"] = "RL (PPO)"
    df_base["Agent"] = "Baseline"
    
    df = pd.concat([df_rl, df_base], ignore_index=True)
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Market Price (Real Data)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="market_price", hue="Agent")
    plt.title("Real Data: Market Price Stability (1 Week)")
    plt.savefig("real_market_price.png")
    
    # Plot 2: Cumulative Reward
    df_rl["cum_reward"] = df_rl["total_reward"].cumsum()
    df_base["cum_reward"] = df_base["total_reward"].cumsum()
    df_cum = pd.concat([df_rl, df_base], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_cum, x="step", y="cum_reward", hue="Agent")
    plt.title("Real Data: Cumulative Social Welfare")
    plt.savefig("real_cumulative_reward.png")
    
    print("Saved plots.")

if __name__ == "__main__":
    plot_real()
