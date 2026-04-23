
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from scipy.stats import linregress

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust
from train.run_v7_retraining import VectorizedMultiAgentEnv

RESULTS_DIR = "research_q1/results/emergence_plots_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def generate_publication_plots():
    # 1. Setup Environment and Model
    n = 4
    model_path = "models_v7_emergence/ppo_n4_emergence"
    norm_path = "models_v7_emergence/vec_normalize_n4_emergence.pkl"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model {model_path} not found.")
        return

    csv_path = "scenarios/scenario_high_p2p.csv"
    
    eval_env = VectorizedMultiAgentEnv(n_agents=n)
    eval_env.env.data_file = csv_path
    eval_env.env._load_data()
    eval_env.env.random_start_day = False
    
    if os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = True

    model = PPO.load(model_path)
    
    steps = 168 # 1 week
    obs = eval_env.reset()
    metrics = []
    
    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        info = infos[0]
        
        metrics.append({
            'step': t,
            'p2p_t': info.get('p2p_volume', 0),
            'grid_import_t': info.get('total_import', 0),
            'clean_profit_t': info.get('mean_clean_profit', 0),
            'success_t': info.get('success_rate', 0),
            'battery_usage_t': info.get('total_battery_usage', 0),
            'carbon_intensity': info.get('carbon_intensity', 0.4)
        })

    df = pd.DataFrame(metrics)
    window = 12

    # --- PLOT 1: EMERGENCE OF TRADING BEHAVIOR ---
    plt.figure(figsize=(10, 5))
    p2p_roll = normalize(df['p2p_t'].rolling(window=window).mean().fillna(0))
    success_roll = normalize(df['success_t'].rolling(window=window).mean().fillna(0))
    battery_roll = normalize(df['battery_usage_t'].rolling(window=window).mean().fillna(0))
    
    plt.plot(df['step'], p2p_roll, label="P2P Volume (Norm)", linewidth=2, color='tab:blue')
    plt.plot(df['step'], success_roll, label="Success Rate (Norm)", linewidth=2, color='tab:orange')
    plt.plot(df['step'], battery_roll, label="Battery Usage (Norm)", linewidth=2, color='tab:green', linestyle='--')
    
    plt.xlabel('Time (Hour)', fontweight='bold')
    plt.ylabel('Normalized Magnitude [0,1]', fontweight='bold')
    plt.title('Emergence of Decentralized Trading Behavior', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot1_emergence.png"), dpi=300)
    plt.close()

    # --- PLOT 2: P2P vs GRID DEPENDENCY (MECHANISM) ---
    plt.figure(figsize=(6, 5))
    mask = df['p2p_t'] > 0.05
    if mask.any():
        x = df.loc[mask, 'p2p_t']
        y = df.loc[mask, 'grid_import_t']
        
        plt.scatter(x, y, alpha=0.7, color='tab:purple', edgecolors='black')
        
        # Regression
        m, b, r, p, std = linregress(x, y)
        plt.plot(x, m*x + b, color='darkred', linewidth=2, label=f'Trend (Slope={m:.2f})')
        
        plt.xlabel('P2P Traded Energy (kW)', fontweight='bold')
        plt.ylabel('Grid Import (kW)', fontweight='bold')
        plt.title('P2P Trading vs Grid Dependency (Filtered)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
    else:
        plt.text(0.5, 0.5, "Insufficient P2P Volume (>0.05) for regression", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot2_mechanism.png"), dpi=300)
    plt.close()

    # --- PLOT 3: PROFIT vs CO2 (OUTCOME) ---
    plt.figure(figsize=(10, 5))
    df['co2_t'] = df['grid_import_t'] * df['carbon_intensity']
    profit_s = df['clean_profit_t'].rolling(window=window).mean()
    co2_s = df['co2_t'].rolling(window=window).mean()
    
    # Valid indices after rolling
    valid = ~profit_s.isna()
    x = profit_s[valid]
    y = co2_s[valid]
    
    # Shift profit
    x = x - x.min()
    
    sc = plt.scatter(x, y, c=df.loc[valid, 'p2p_t'], cmap='viridis', alpha=0.7, edgecolors='black')
    cbar = plt.colorbar(sc)
    cbar.set_label('P2P Volume (kW)', fontweight='bold')
    
    # Regression
    m, b, r, p, std = linregress(x, y)
    plt.plot(x, m*x + b, color='red', linewidth=2, label=f'R = {r:.3f}')
    
    plt.xlabel('Shifted Average Profit ($)', fontweight='bold')
    plt.ylabel('CO2 Emissions (kg)', fontweight='bold')
    plt.title('Profit vs CO2 Emissions (Short-Term Dynamics)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot3_outcome.png"), dpi=300)
    plt.close()

    print(f"Publication-quality plots generated in {RESULTS_DIR}")

if __name__ == "__main__":
    generate_publication_plots()
