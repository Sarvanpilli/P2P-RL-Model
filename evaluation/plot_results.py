import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(csv_path, output_dir="evaluation"):
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Market Dynamics: Price vs Net Imbalance
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["market_price"], label="Market Price ($/kWh)", color="blue", linewidth=2)
    plt.xlabel("Step (Hour)")
    plt.ylabel("Price ($/kWh)")
    plt.title("Market Price Dynamics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "market_prices.png"))
    plt.close()
    
    # 2. Grid Interaction (grid_flow = export - import in kW)
    grid_flow = df["grid_flow"] if "grid_flow" in df.columns else (df["total_export"] - df["total_import"])
    plt.figure(figsize=(10, 6))
    plt.fill_between(df["step"], grid_flow, 0, where=(grid_flow > 0), color='green', alpha=0.3, label='Export')
    plt.fill_between(df["step"], grid_flow, 0, where=(grid_flow < 0), color='red', alpha=0.3, label='Import')
    plt.plot(df["step"], grid_flow, color="black", linewidth=1)
    plt.xlabel("Step (Hour)")
    plt.ylabel("Power (kW) [+Export / -Import]")
    plt.title("Grid Interaction (Net Flow)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "grid_flow.png"))
    plt.close()
    
    # 3. Agent SoC
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        if "soc" in col and "agent" in col:
            agent_id = col.split("_")[1]
            plt.plot(df["step"], df[col], label=f"Agent {agent_id}", linewidth=1.5)
    plt.xlabel("Step (Hour)")
    plt.ylabel("SoC (kWh)")
    plt.title("Battery State of Charge")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "agent_soc.png"))
    plt.close()
    
    # 4. CO2 Impact & Cumulative Profit
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Step (Hour)')
    ax1.set_ylabel('Cumulative CO2 (kg)', color=color)
    cum_co2 = df["cumulative_carbon_kg"] if "cumulative_carbon_kg" in df.columns else (df["total_carbon_kg"].cumsum() if "total_carbon_kg" in df.columns else None)
    if cum_co2 is not None:
        ax1.plot(df["step"], cum_co2, color=color, linewidth=2, label="CO2")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative Profit ($)', color=color)  # we already handled the x-label with ax1
    # Calculate total profit measure
    profit_cols = [c for c in df.columns if "profit" in c]
    total_profit_step = df[profit_cols].sum(axis=1)
    ax2.plot(df["step"], total_profit_step.cumsum(), color=color, linewidth=2, linestyle='--', label="Profit")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Environmental vs Economic Performance")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(output_dir, "impact_analysis.png"))
    plt.close()

    # 5. Guard Interventions (Robustness Check)
    if "guard_intervention" in df.columns:
        plt.figure(figsize=(10, 4))
        interventions = df[df["guard_intervention"] > 0]
        plt.scatter(interventions["step"], np.ones(len(interventions)), color='red', marker='x', s=100, label="Safety Breach Prevented")
        plt.xlim(0, max(df["step"]))
        plt.ylim(0, 2)
        plt.yticks([])
        plt.xlabel("Step (Hour)")
        plt.title("Safety Guard Interventions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "safety_interventions.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    CSV_PATH = "evaluation/evaluation_results.csv"
    plot_results(CSV_PATH)
