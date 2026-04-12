
import os
import sys
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust

# CONFIG
STEPS_PER_EXP = 30000
EVAL_STEPS = 500
RESULTS_DIR = "research_q1/results/incentive_experiments"
MODEL_DIR = "models_incentive"
OS_WINDOWS = os.name == 'nt'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(beta=1.0):
    def _init():
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
        )
        env.beta = beta # Inject beta
        return env
    return _init

def evaluate_beta(model_path, vec_path, beta, steps=EVAL_STEPS):
    print(f"Evaluating Beta={beta} model...")
    
    # Use baseline to check comparison
    eval_env = DummyVecEnv([make_env(beta=beta)])
    eval_env = VecNormalize.load(vec_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    model = PPO.load(model_path)
    obs = eval_env.reset()
    
    metrics = {
        'profit': 0.0,
        'p2p_vol': 0.0,
        'grid_imp': 0.0,
        'demand': 0.0,
        'co2': 0.0
    }
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        i = info[0]
        metrics['profit'] += i.get('market_profit_usd', 0)
        metrics['p2p_vol'] += i.get('p2p_volume_kwh_step', 0)
        metrics['grid_imp'] += i.get('grid_import_kwh', 0)
        metrics['demand'] += i.get('total_demand_kw', 1e-6)
        metrics['co2'] += i.get('grid_import_kwh', 0) * i.get('carbon_intensity', 0.5)
        
    res = {
        'Beta': beta,
        'avg_profit': metrics['profit'] / steps,
        'p2p_volume': metrics['p2p_vol'],
        'grid_dependency': (metrics['grid_imp'] / metrics['demand']) * 100.0,
        'carbon_kg': metrics['co2']
    }
    return res

def run_experiment(beta_val):
    print(f"\n>>> STARTING EXPERIMENT: Beta = {beta_val}")
    
    # Init Env
    n_envs = 4
    train_env = SubprocVecEnv([make_env(beta=beta_val) for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=STEPS_PER_EXP)
    
    model_path = os.path.join(MODEL_DIR, f"ppo_beta_{beta_val}")
    vec_path = os.path.join(MODEL_DIR, f"vec_beta_{beta_val}.pkl")
    model.save(model_path)
    train_env.save(vec_path)
    
    result = evaluate_beta(model_path, vec_path, beta_val)
    train_env.close()
    return result

if __name__ == "__main__":
    # 1. BASELINE (Simulated current state or run new one if path missing)
    # For speed, let's treat Beta=0.35 as current baseline if needed
    
    betas = [1.0, 1.5, 2.0]
    all_results = []
    
    # Add manual baseline result (Approximated from current known stats if necessary)
    # But let's run a quick 0.35 baseline for scientific consistency
    betas = [0.35] + betas 
    
    for b in betas:
        res = run_experiment(b)
        all_results.append(res)
        
    # Generate Table
    df = pd.DataFrame(all_results)
    
    # Calculate Grid Reduction % vs Baseline (Beta=0.35)
    baseline_dep = df.loc[df['Beta'] == 0.35, 'grid_dependency'].values[0]
    df['Grid_Reduction_%'] = (baseline_dep - df['grid_dependency']) / baseline_dep * 100
    
    print("\n--- FINAL COMPARISON TABLE ---")
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(RESULTS_DIR, "incentive_comparison.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/incentive_comparison.csv")
