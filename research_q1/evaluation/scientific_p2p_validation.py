
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train.energy_env_robust import EnergyMarketEnvRobust

def run_validation():
    # 1. Setup Env (4-agent fixed system)
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=False, # Use fixed start for validation
        forecast_horizon=4
    )
    
    # 2. Load Model 
    # Try to find a trained hybrid model, fallback to Random Policy if none
    model_path = "models_phase5_hybrid/ppo_hybrid_final.zip"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
        is_random = False
    else:
        print("WARNING: No trained model found. Running Random Policy baseline.")
        is_random = True
        
    # 3. Episode Run
    obs, _ = env.reset()
    done = False
    truncated = False
    
    history = []
    
    while not (done or truncated):
        if is_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, done, truncated, info = env.step(action)
        
        # Collect Metrics
        history.append({
            't': env.timestep_count,
            'p2p_volume': info.get('p2p_volume_kwh_step', 0),
            'grid_import': info.get('total_import', 0),
            'grid_export': info.get('total_export', 0),
            'profit': info.get('absolute_profit_usd', 0),
            'co2': info.get('co2_penalty', 0) / 0.10, # Reverse coeff to get kg/step approx
            'success_rate': env.market_history['last_success_rate']
        })
        
    df = pd.DataFrame(history)
    df['cumulative_p2p'] = df['p2p_volume'].cumsum()
    df['cumulative_profit'] = df['profit'].cumsum()
    df['cumulative_co2'] = df['co2'].cumsum()
    
    # 4. Generate Publication Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: P2P Volume
    axes[0, 0].plot(df['t'], df['cumulative_p2p'], color='blue', lw=2)
    axes[0, 0].set_title("1. Cumulative P2P Trading Volume")
    axes[0, 0].set_ylabel("kWh")
    
    # Plot 2: Grid Import
    axes[0, 1].plot(df['t'], df['grid_import'], color='orange', alpha=0.5)
    axes[0, 1].plot(df['t'], df['grid_import'].rolling(12).mean(), color='red', lw=2)
    axes[0, 1].set_title("2. Grid Import (Moving Average)")
    axes[0, 1].set_ylabel("kW")
    
    # Plot 3: Profit vs CO2 (Dual Axis)
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    p1 = ax3.plot(df['t'], df['cumulative_profit'], color='green', label='Profit (USD)', lw=2)
    p2 = ax3_twin.plot(df['t'], df['cumulative_co2'], color='gray', linestyle='--', label='CO2 (kg)')
    ax3.set_title("3. Cumulative Profit vs CO2")
    ax3.set_ylabel("USD")
    ax3_twin.set_ylabel("kg")
    
    # Plot 4: Success Rate
    axes[1, 1].plot(df['t'], df['success_rate'], color='purple', lw=2)
    axes[1, 1].set_title("4. Market Trade Success Rate")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plot_path = "research_q1/results/scientific_upgrade_validation.png"
    plt.savefig(plot_path)
    print(f"Validation plots saved to {plot_path}")
    
    # Summary Table
    total_p2p = df['p2p_volume'].sum()
    final_profit = df['cumulative_profit'].iloc[-1]
    
    print("\n" + "="*30)
    print("UPGRADE VALIDATION SUMMARY")
    print("="*30)
    print(f"Total P2P Traded:   {total_p2p:.2f} kWh")
    print(f"Final Profit:       {final_profit:.2f} USD")
    print(f"Avg Success Rate:   {df['success_rate'].mean():.2%}")
    print("="*30)

if __name__ == "__main__":
    run_validation()
