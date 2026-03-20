import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path="research_q1/results/results_rl_vs_baseline.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluate script first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Smooth data for cleaner plotting
    window = 24 # 24 hour rolling average
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # 1. P2P Volume
    axes[0].plot(df['rl_p2p_volume'].rolling(window).mean(), label='RL Agent (SLIM)', color='blue', linewidth=2)
    axes[0].plot(df['base_p2p_volume'].rolling(window).mean(), label='Baseline (Grid Only)', color='red', linestyle='--', linewidth=2)
    axes[0].set_title('P2P Traded Volume over Time (24h Rolling Mean)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('kWh Traded', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Cumulative Profit
    axes[1].plot(df['rl_profit'], label='RL Agent (SLIM)', color='green', linewidth=2)
    axes[1].plot(df['base_profit'], label='Baseline (Grid Only)', color='orange', linestyle='--', linewidth=2)
    axes[1].set_title('Cumulative Economic Profit', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Profit ($)', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Grid Import
    axes[2].plot(df['rl_grid_import'].rolling(window).mean(), label='RL Agent Import', color='purple', linewidth=2)
    axes[2].plot(df['base_grid_import'].rolling(window).mean(), label='Baseline Import', color='brown', linestyle='--', linewidth=2)
    axes[2].set_title('Grid Dependency (Import Volume over Time)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('kWh Imported', fontsize=12)
    axes[2].set_xlabel('Timestep (Hours)', fontsize=12)
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = "research_q1/results/market_performance.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"Plot correctly generated and saved to {out_file}")

if __name__ == "__main__":
    plot_results()
