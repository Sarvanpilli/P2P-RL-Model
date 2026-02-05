import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration (use paths relative to project root or this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(_SCRIPT_DIR, "evaluation_results.csv")
DATA_PATH = os.path.join(_SCRIPT_DIR, "ausgrid_p2p_energy_dataset.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

def get_tou_prices(hour):
    """Returns (retail, feed_in) for a given hour."""
    # From energy_env_robust.py
    if 0 <= hour < 6 or hour >= 23:
        return 0.10, 0.03 # Off-Peak
    elif 17 <= hour < 21:
        return 0.35, 0.15 # Peak
    else:
        return 0.20, 0.08 # Standard

def load_data():
    print("Loading data...")
    rl_df = pd.read_csv(RESULTS_PATH)
    raw_df = pd.read_csv(DATA_PATH)
    return rl_df, raw_df

def plot_soc(rl_df):
    print("Generating Plot 1: SoC vs Time...")
    plt.figure()
    
    # Plot only first 300 steps (approx 12 days) for clarity, or an aggregate?
    # User asked for "Time (hours or days)" showing bounds.
    # If we plot 8000 points it's messy. Let's plot a 1-week window (168 hours) from a busy period.
    # Or plot the whole thing with transparency.
    
    subset = rl_df.iloc[:168*2] # 2 Weeks
    
    soc_cols = [c for c in rl_df.columns if 'soc' in c]
    for col in soc_cols:
        plt.plot(subset['step'], subset[col], label=col, alpha=0.8)
        
    plt.axhline(0, color='red', linestyle='--', label='Min Limit (0)')
    plt.axhline(50, color='red', linestyle='--', label='Max Limit (50)')
    
    plt.xlabel("Hour")
    plt.ylabel("State of Charge (kWh)")
    plt.title("Battery State of Charge (SoC) - 2 Week Sample")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_1_soc.png"))
    plt.close()

def plot_grid_flow(rl_df):
    print("Generating Plot 2: Grid Import/Export...")
    plt.figure()
    
    subset = rl_df.iloc[:168] # 1 Week
    
    plt.plot(subset['step'], subset['total_import'], label='Grid Import', color='red', alpha=0.7)
    plt.plot(subset['step'], -subset['total_export'], label='Grid Export', color='green', alpha=0.7)
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel("Hour")
    plt.ylabel("Power (kW) [+Imp / -Exp]")
    plt.title("Grid Interaction (Community Aggregate) - 1 Week Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_2_grid_flow.png"))
    plt.close()

def plot_market_price(rl_df):
    print("Generating Plot 3: Market Price Analysis...")
    plt.figure()
    
    # Calculate average price per hour of day
    rl_df['hour_of_day'] = rl_df['step'] % 24
    avg_price = rl_df.groupby('hour_of_day')['market_price'].mean()
    
    # Bounds
    hours = np.arange(24)
    retail = [get_tou_prices(h)[0] for h in hours]
    feedin = [get_tou_prices(h)[1] for h in hours]
    
    plt.plot(hours, avg_price, 'o-', label='P2P Clearing Price', color='blue', linewidth=2)
    plt.step(hours, retail, where='mid', linestyle='--', label='Grid Retail Price', color='red', alpha=0.5)
    plt.step(hours, feedin, where='mid', linestyle='--', label='Grid Feed-in Price', color='green', alpha=0.5)
    
    plt.xlabel("Hour of Day")
    plt.ylabel("Price ($/kWh)")
    plt.title("Average Market Clearing Price vs Grid Tariffs")
    plt.xticks(hours)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_3_market_price.png"))
    plt.close()

def plot_profit_comparison(rl_df, raw_df):
    print("Generating Plot 4: Cumulative Profit...")
    plt.figure()
    
    # 1. Calculate Baseline Cumulative Profit
    baseline_profit = []
    current_profit = 0.0
    
    ids = [0, 1, 2, 3]
    # Ensure lengths match
    min_len = min(len(rl_df), len(raw_df))
    
    print(f"Calculating financial cumulative series for {min_len} steps...")
    
    for i in range(min_len):
        hour = i % 24
        retail, feed_in = get_tou_prices(hour)
        
        # Baseline Step
        step_cost = 0.0
        for agent_id in ids:
            dem = raw_df.loc[i, f'agent_{agent_id}_demand_kw']
            pv = raw_df.loc[i, f'agent_{agent_id}_pv_kw']
            net = dem - pv
            
            if net > 0:
                step_cost += net * retail
            else:
                step_cost -= abs(net) * feed_in
        
        # Profit = -Cost
        current_profit += (-step_cost)
        baseline_profit.append(current_profit)
        
    # 2. Calculate RL Cumulative Profit
    rl_cum_profit = []
    current_rl = 0.0
    
    for i in range(min_len):
        hour = i % 24
        retail, feed_in = get_tou_prices(hour)
        
        imp = rl_df.loc[i, 'total_import']
        exp = rl_df.loc[i, 'total_export']
        
        # Cost
        step_cost = (imp * retail) - (exp * feed_in)
        current_rl += (-step_cost)
        rl_cum_profit.append(current_rl)
        
    # Plot
    steps = np.arange(min_len)
    plt.plot(steps, baseline_profit, label='Grid-Only Baseline (No Battery)', linestyle='--', color='gray')
    plt.plot(steps, rl_cum_profit, label='P2P-RL System', color='green', linewidth=2)
    
    # Highlight final value
    plt.text(steps[-1], baseline_profit[-1], f"${baseline_profit[-1]:.0f}", ha='left', va='center')
    plt.text(steps[-1], rl_cum_profit[-1], f"${rl_cum_profit[-1]:.0f}", ha='left', va='center', fontweight='bold', color='green')
    
    plt.xlabel("Simulation Hour")
    plt.ylabel("Cumulative Community Profit ($)")
    plt.title("Economic Value: P2P-RL vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_4_cumulative_profit.png"))
    plt.close()

if __name__ == "__main__":
    rl, raw = load_data()
    plot_soc(rl)
    plot_grid_flow(rl)
    plot_market_price(rl)
    plot_profit_comparison(rl, raw)
    print("Plots generated in evaluation/plots/")
