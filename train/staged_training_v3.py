
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# CONFIGURATION
PHASES = [10000, 20000, 50000, 100000] # Cumulative steps
EVAL_STEPS = 500
RESULTS_DIR = "research_q1/results/staged_training"
MODEL_DIR = "models_staged"
LOG_FILE = os.path.join(RESULTS_DIR, "training_logs.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
BASELINE_FILE = "research_q1/results/carbon_baseline.json"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env():
    return EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=True,
    )

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

def evaluate_model(model_path, vec_path, steps=EVAL_STEPS):
    print(f"Evaluating model: {model_path}...")
    
    # Init Env for evaluation (Must be VecEnv for VecNormalize)
    eval_env = DummyVecEnv([make_env])
    # Apply normalization (no training)
    eval_env = VecNormalize.load(vec_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    model = PPO.load(model_path)
    
    # Load Carbon Baseline
    baseline_val = 0.1
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline_val = json.load(f).get('avg_co2_per_step', 0.1)

    obs = eval_env.reset()
    metrics = {
        'clean_profit': 0.0,
        'economic_profit': 0.0,
        'p2p_volume': 0.0,
        'grid_import': 0.0,
        'total_demand': 0.0,
        'co2_emissions': 0.0,
        'trade_success': 0
    }
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        # Info is a list for VecEnv
        i = info[0]
        metrics['clean_profit'] += i.get('market_profit_usd', 0)
        metrics['economic_profit'] += i.get('economic_profit_usd', 0)
        metrics['p2p_volume'] += i.get('p2p_volume_kwh_step', 0)
        metrics['grid_import'] += i.get('grid_import_kwh', 0)
        metrics['total_demand'] += i.get('total_demand_kw', 1e-6)
        metrics['co2_emissions'] += i.get('grid_import_kwh', 0) * i.get('carbon_intensity', 0.5)
        if i.get('p2p_volume_kwh_step', 0) > 0:
            metrics['trade_success'] += 1
            
    # Final Averages/Percentages
    results = {
        'avg_market_profit': metrics['clean_profit'] / steps,
        'cumulative_clean_profit': metrics['clean_profit'],
        'cumulative_economic_profit': metrics['economic_profit'],
        'cumulative_p2p_volume': metrics['p2p_volume'],
        'grid_dependency': (metrics['grid_import'] / (metrics['total_demand'] + 1e-9)) * 100.0,
        'carbon_reduction': (1.0 - (metrics['co2_emissions'] / (baseline_val * steps + 1e-9))) * 100.0,
        'trade_success_rate': (metrics['trade_success'] / steps) * 100.0
    }
    return results

def run_staged_training():
    print("Starting Staged Training Scaling (v3)...")
    
    history = []
    best_clean_profit = -np.inf
    last_phase_profit = -np.inf
    stagnant_count = 0
    best_model_data = {}

    # Initial Env setup
    n_envs = 4
    train_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    model = PPO("MlpPolicy", train_env, verbose=0, device='auto')
    
    current_steps = 0
    for i, target_steps in enumerate(PHASES):
        phase_id = i + 1
        steps_to_train = target_steps - current_steps
        print(f"\n--- PHASE {phase_id}: Training to {target_steps} steps ---")
        
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        current_steps = target_steps
        
        # Save Checkpoint
        phase_model_path = os.path.join(MODEL_DIR, f"ppo_phase_{phase_id}")
        phase_vec_path = os.path.join(MODEL_DIR, f"vec_phase_{phase_id}.pkl")
        model.save(phase_model_path)
        train_env.save(phase_vec_path)
        
        # Evaluate
        # We use the raw file path for evaluate_model to ensure clean reload
        results = evaluate_model(phase_model_path, phase_vec_path)
        results['total_steps'] = current_steps
        results['phase'] = phase_id
        
        print(f"Results: Profit=${results['avg_market_profit']:.4f} | Grid_Dep={results['grid_dependency']:.2f}% | CO2_Red={results['carbon_reduction']:.2f}%")
        
        history.append(results)
        
        # Select Best Model
        if results['avg_market_profit'] > best_clean_profit:
            best_clean_profit = results['avg_market_profit']
            best_model_data = {
                'phase': phase_id,
                'path': phase_model_path,
                'vec_path': phase_vec_path
            }
            print(f"New Best Model Found in Phase {phase_id}!")
            # Copy to best
            model.save(os.path.join(MODEL_DIR, "best_model"))
            train_env.save(os.path.join(MODEL_DIR, "best_vec_normalize.pkl"))

        # Early Stopping Check
        # If profit doesn't improve for 2 consecutive phases
        if results['avg_market_profit'] <= last_phase_profit:
            stagnant_count += 1
        else:
            stagnant_count = 0
            
        last_phase_profit = results['avg_market_profit']
        
        if stagnant_count >= 2:
            print(f"EARLY STOPPING: No profit improvement for {stagnant_count} phases. Optimal training reached.")
            break
            
        # Overtraining Check (Optional Step 5)
        # If dependency starts increasing significantly
        if i > 0 and results['grid_dependency'] > history[i-1]['grid_dependency'] * 1.1:
            print(f"OVERTRAINING DETECTED: Grid dependency increased from {history[i-1]['grid_dependency']:.2f}% to {results['grid_dependency']:.2f}%. Stopping.")
            break

    # Save Logs
    df = pd.DataFrame(history)
    df.to_csv(LOG_FILE, index=False)
    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining Finished. Best model from Phase {best_model_data['phase']}.")
    
    # Final Validation (Step 8)
    print("\n--- Running Final Validation (2000 steps) ---")
    final_results = evaluate_model(
        os.path.join(MODEL_DIR, "best_model"), 
        os.path.join(MODEL_DIR, "best_vec_normalize.pkl"), 
        steps=2000
    )
    print(f"Final Validation Result: {final_results}")
    
    # Step 7: Generate Convergence Plots
    generate_plots(df)

def generate_plots(df):
    print("Generating Convergence Plots...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 1. Clean Profit
    axes[0].plot(df['total_steps'], df['avg_market_profit'], marker='o', color='green', label='Avg Market Profit')
    axes[0].set_title("Market Profit vs Training Steps")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("USD / Step")
    axes[0].grid(True)
    
    # 2. Grid Dependency
    axes[1].plot(df['total_steps'], df['grid_dependency'], marker='s', color='blue', label='Grid Dependency %')
    axes[1].set_title("Grid Dependency vs Training Steps")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Dependency %")
    axes[1].grid(True)
    
    # 3. P2P Volume
    axes[2].plot(df['total_steps'], df['cumulative_p2p_volume'], marker='^', color='orange', label='Cum P2P Volume')
    axes[2].set_title("P2P Volume vs Training Steps")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("kWh")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_plots.png"))
    print(f"Plots saved to {RESULTS_DIR}/convergence_plots.png")

if __name__ == "__main__":
    run_staged_training()
