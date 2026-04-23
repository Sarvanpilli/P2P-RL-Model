
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

RESULTS_FILE = "evaluation/7day_benchmark_results.json"
OUTPUT_DIR = "evaluation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def plot_time_series(data):
    """168-hour time series for Grid and P2P."""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    for name, result in data.items():
        df = pd.DataFrame(result)
        plt.plot(df["hour"], df["grid_dependency"], label=name, alpha=0.7)
    plt.title("Grid Dependency Over 168-Hour Horizon (%)")
    plt.ylabel("Dependency (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    for name, result in data.items():
        df = pd.DataFrame(result)
        plt.plot(df["hour"], df["p2p_volume_step"], label=name, alpha=0.7)
    plt.title("P2P Trading Volume (kWh/Step)")
    plt.ylabel("Volume (kWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    for name, result in data.items():
        df = pd.DataFrame(result)
        plt.plot(df["hour"], df["soc"], label=name, alpha=0.7, linestyle='--')
    plt.title("Community Battery SoC (%)")
    plt.ylabel("SoC (%)")
    plt.xlabel("Hours")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeseries_168h.png"))
    plt.close()

def plot_comparative_bars(data):
    """Comparative bar charts for key KPIs."""
    avg_results = {}
    
    for name, result in data.items():
        df = pd.DataFrame(result)
        avg_results[name] = {
            "success_rate": df["success_rate"].mean(),
            "grid_dependency": df["grid_dependency"].mean(),
            "profit": df["clean_profit_usd_step"].sum() 
        }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    names = list(avg_results.keys())
    colors = ['#2E86C1', '#707B7C', '#F39C12', '#27AE60']
    
    # Success Rate
    vals = [avg_results[n]["success_rate"] for n in names]
    axes[0].bar(names, vals, color=colors)
    axes[0].set_title("Avg Trade Success Rate (%)")
    axes[0].set_ylabel("%")
    
    # Grid Dependency
    vals = [avg_results[n]["grid_dependency"] for n in names]
    axes[1].bar(names, vals, color=colors)
    axes[1].set_title("Avg Grid Dependency (%)")
    axes[1].set_ylabel("%")
    
    # Cumulative Profit
    vals = [avg_results[n]["profit"] for n in names]
    axes[2].bar(names, vals, color=colors)
    axes[2].set_title("Total Weekly Community Savings ($)")
    axes[2].set_ylabel("USD")
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_bars.png"))
    plt.close()

def plot_cumulative_plots(data):
    """Cumulative plots for Volume and CO2."""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for name, result in data.items():
        df = pd.DataFrame(result)
        plt.plot(df["hour"], df["p2p_volume_step"].cumsum(), label=name)
    plt.title("Cumulative P2P energy flow (kWh)")
    plt.xlabel("Hours")
    plt.ylabel("kWh")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, result in data.items():
        df = pd.DataFrame(result)
        plt.plot(df["hour"], df["co2_kg_step"].cumsum(), label=name)
    plt.title("Cumulative Carbon Footprint (kg CO2)")
    plt.xlabel("Hours")
    plt.ylabel("kg")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_impact.png"))
    plt.close()

def main():
    if not os.path.exists(RESULTS_FILE):
        return
    data = load_data()
    plot_time_series(data)
    plot_comparative_bars(data)
    plot_cumulative_plots(data)
    print(f"Science plots generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
