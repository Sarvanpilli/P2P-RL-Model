
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

RESULTS_DIR = "research_q1/results/emergence_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_plots():
    # 1. Setup Environment and Model
    n = 4
    model_path = "models_v7_emergence/ppo_n4_emergence"
    norm_path = "models_v7_emergence/vec_normalize_n4_emergence.pkl"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model {model_path} not found.")
        return

    # Using the high P2P scenario for scientific depth
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
    
    # 2. Run Simulation (1 week)
    steps = 168
    obs = eval_env.reset()
    metrics = []
    
    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        info = infos[0]
        
        metrics.append({
            'step': t,
            'p2p_volume': info.get('p2p_volume', 0),
            'grid_import': info.get('total_import', 0),
            'profit': info.get('mean_clean_profit', 0),
            'co2': info.get('total_import', 0) * info.get('carbon_intensity', 0.4),
            'success': float(info.get('p2p_volume', 0) > 1e-3),
            'battery_usage': info.get('total_battery_usage', 0)
        })

    df = pd.DataFrame(metrics)
    
    # --- PLOT 1: Profit vs CO2 Emissions ---
    plt.figure(figsize=(10, 6))
    window = 6
    profit_s = df['profit'].rolling(window=window).mean()
    co2_s = df['co2'].rolling(window=window).mean()
    
    # Valid indices after rolling
    valid = ~profit_s.isna()
    
    plt.scatter(profit_s[valid], co2_s[valid], c=df['p2p_volume'][valid], cmap='viridis', alpha=0.6)
    cbar = plt.colorbar()
    cbar.set_label('P2P Volume (kW)')
    
    # Regression
    slope, intercept, r_value, p_value, std_err = linregress(profit_s[valid], co2_s[valid])
    plt.plot(profit_s[valid], intercept + slope*profit_s[valid], 'r', label=f'R={r_value:.2f}')
    
    plt.xlabel('Average Hourly Profit ($)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.title('Profit vs CO2 Emissions (Short-Term Dynamics)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "profit_vs_co2_emergence.png"))
    plt.close()

    # --- PLOT 2: P2P Trading vs Grid Dependency ---
    plt.figure(figsize=(10, 6))
    grid_t = df['grid_import']
    p2p_t = df['p2p_volume']
    
    plt.scatter(p2p_t, grid_t, alpha=0.5, color='blue')
    
    # Regression
    slope, intercept, r_value, p_value, std_err = linregress(p2p_t, grid_t)
    plt.plot(p2p_t, intercept + slope*p2p_t, 'darkred', label=f'Slope={slope:.2f}')
    
    plt.xlabel('P2P Trading Volume (kW)')
    plt.ylabel('Grid Import (kW)')
    plt.title('P2P Trading vs Grid Dependency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "p2p_vs_grid_emergence.png"))
    plt.close()

    # --- PLOT 3: Emergence of Decentralized Trading Behavior ---
    plt.figure(figsize=(12, 6))
    window = 12
    p2p_roll = df['p2p_volume'].rolling(window=window).mean()
    success_roll = df['success'].rolling(window=window).mean()
    battery_roll = df['battery_usage'].rolling(window=window).mean()
    
    plt.plot(df['step'], p2p_roll, label="P2P Volume", linewidth=2)
    plt.plot(df['step'], success_roll, label="Success Rate", linewidth=2)
    plt.plot(df['step'], battery_roll, label="Battery Usage", linewidth=2, linestyle='--')
    
    plt.xlabel('Time (Hour)')
    plt.ylabel('Magnitude')
    plt.title('Emergence of Decentralized Trading Behavior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "trading_behavior_emergence.png"))
    plt.close()

    print(f"Scientific plots generated successfully in {RESULTS_DIR}")

if __name__ == "__main__":
    generate_plots()
