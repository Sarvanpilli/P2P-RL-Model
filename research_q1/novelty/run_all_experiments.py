import os
import pandas as pd
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import sys

# ==========================================
# GLOBAL PARALLELISM CONFIG
# Total CPU cores available: 16
# N_ENVS workers collect rollouts in parallel — each on its own core.
# Total rollout per update = N_ENVS * n_steps (set below)
# ==========================================
N_ENVS = 8  # Use 8 of 16 cores for training (leave 8 for OS + GNN + evaluation)

# Ensure module path is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

# Try importing the GNN policy if it exists, otherwise fallback to MLP
try:
    from research_q1.novelty.gnn_policy import CTDEGNNPolicy
    HAS_GNN = True
except ImportError:
    HAS_GNN = False
    print("WARNING: CTDEGNNPolicy not found. gnn_model will fallback to MlpPolicy.")

# ==========================================
# 1. EXPERIMENT CONFIGURATIONS
# ==========================================
experiments = {
    "baseline_grid": {
        "market": "dynamic",  # Doesn't matter, model is none
        "model": "none",
        "alpha": 0.01,
        "base_delta": 0.03
    },
    "auction_old": {
        "market": "auction",  # Forces delta=0, mid-price clearing
        "model": "ppo",
        "alpha": 0.01,
        "base_delta": 0.03
    },
    "new_market": {
        "market": "dynamic",
        "model": "ppo",
        "alpha": 0.01,
        "base_delta": 0.03
    },
    "no_p2p_reward": {
        "market": "dynamic",
        "model": "ppo",
        "alpha": 0.0,       # Turn off P2P reward shaping
        "base_delta": 0.03
    },
    "gnn_model": {
        "market": "dynamic",
        "model": "gnn" if HAS_GNN else "ppo", # Use GNN if available
        "alpha": 0.01,
        "base_delta": 0.03
    }
}

# ==========================================
# 2. EVALUATION HELPER
# ==========================================
def evaluate_model(model, config, seed, steps=500):
    """Creates a FRESH, non-normalized env for evaluation to avoid VecNormalize stat bias."""
    def make_eval_env():
        return EnergyMarketEnvSLIM(
            n_agents=4, 
            data_file="fixed_training_data.csv",
            enable_safety=True,
            enable_p2p=True,
            market_type=config["market"],
            alpha_p2p=config.get("alpha", 0.005),
            base_delta=config.get("base_delta", 0.03),
            seed=seed + 100  # Different seed from training to avoid overfitting to a specific sequence
        )

    eval_env = DummyVecEnv([make_eval_env])
    obs = eval_env.reset()
    
    metrics = {"p2p_volume": [], "profit": [], "grid_import": [], "grid_export": []}
    
    for t in range(steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Baseline: zero actions (agents do not use battery or P2P, pure grid-only)
            action = np.zeros((1, eval_env.envs[0].action_space.shape[0]))
            
        obs, rewards, dones, infos = eval_env.step(action)
        info = infos[0]
        
        metrics["p2p_volume"].append(info["p2p_volume"])
        metrics["profit"].append(info["profit"])
        metrics["grid_import"].append(info["grid_import"])
        metrics["grid_export"].append(info["grid_export"])
        
    eval_env.close()
        
    return {
        "profit": metrics["profit"][-1],           # Cumulative by end of episode
        "p2p_volume": np.sum(metrics["p2p_volume"]),
        "grid_import": np.sum(metrics["grid_import"]),
        "grid_export": np.sum(metrics["grid_export"]),
        "trace_p2p": metrics["p2p_volume"]
    }

# ==========================================
# 3. TRAINING AND EXECUTION WORKER
# ==========================================
def run_single_experiment(exp_name, config, seed, timesteps=100000, eval_steps=500):
    print(f"\n--- Running Experiment: {exp_name} | Seed: {seed} ---")
    log_dir = f"research_q1/logs/{exp_name}/seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Environment Construction (for training)
    def make_env():
        return EnergyMarketEnvSLIM(
            n_agents=4, 
            data_file="fixed_training_data.csv",
            enable_safety=True,
            enable_p2p=True,
            market_type=config["market"],
            alpha_p2p=config.get("alpha", 0.005),
            base_delta=config.get("base_delta", 0.03),
            seed=seed
        )
        
    # 2. Model Initialization
    model = None
    if config["model"] != "none":
        # SubprocVecEnv: each worker is a separate process with its own copy of the env.
        # This fully utilizes multiple CPU cores during rollout collection.
        env_fns = [make_env for _ in range(N_ENVS)]
        train_env = SubprocVecEnv(env_fns, start_method="spawn")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        policy_class = "MlpPolicy"
        policy_kwargs = {}
        
        if config["model"] == "gnn" and HAS_GNN:
            policy_class = CTDEGNNPolicy
            policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
            
        print(f"Training {config['model']} for {timesteps} timesteps ({N_ENVS} parallel envs)...")
        model = PPO(
            policy_class, 
            train_env, 
            learning_rate=3e-4, 
            n_steps=1024,         # Per-env steps per update. Total = N_ENVS * n_steps = 8192
            batch_size=256,       # Larger batch to match larger rollout buffer
            n_epochs=10, 
            gamma=0.99, 
            gae_lambda=0.95, 
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            seed=seed,
            verbose=0
        )
        model.learn(total_timesteps=timesteps)
        
        # Save model before closing env
        save_path = os.path.join(log_dir, "ppo_model")
        model.save(save_path)
        train_env.close()
    
    # 3. Evaluation on FRESH clean environment (no VecNormalize bias)
    print(f"Evaluating {exp_name} for {eval_steps} timesteps...")
    results = evaluate_model(model, config, seed, steps=eval_steps)
    print(f"  -> Profit: ${results['profit']:.2f} | P2P: {results['p2p_volume']:.2f} kWh")
    
    return results

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
def run_all_experiments():
    seeds = [0, 1, 2, 3, 4]   # Full publication sweep: 5 seeds
    training_steps = 100_000 # Full 100k convergence run
    eval_steps = 500
    
    all_results = []
    
    # Phase 1: Core Experiments Sweep
    for exp_name, config in experiments.items():
        for seed in seeds:
            res = run_single_experiment(exp_name, config, seed, training_steps, eval_steps)
            
            all_results.append({
                "experiment_name": exp_name,
                "seed": seed,
                "delta": config.get("base_delta", 0.03),
                "profit": res["profit"],
                "p2p_volume": res["p2p_volume"],
                "grid_import": res["grid_import"],
                "grid_export": res["grid_export"]
            })
            
    # Phase 2: Delta Sweep Analysis
    print("\n===========================================")
    print("STARTING DELTA SENSITIVITY SWEEP")
    print("===========================================")
    
    delta_values = [0.01, 0.02, 0.03, 0.05]
    delta_config = experiments["new_market"].copy()
    
    all_traces = []
    
    # Extract traces from phase 1 for timeline plotting
    for res in all_results:
        for t, vol in enumerate(res.pop("trace_p2p", [])):
            all_traces.append({
                "experiment_name": res["experiment_name"],
                "seed": res["seed"],
                "timestep": t,
                "p2p_volume": vol
            })
            
    for delta in delta_values:
        # Skip 0.03 as it is already computed in the main sweep above
        if delta == 0.03: continue 
        
        delta_config["base_delta"] = delta
        exp_name = f"delta_sweep_{delta}"
        
        for seed in seeds:
            res = run_single_experiment(exp_name, delta_config, seed, training_steps, eval_steps)
            
            res.pop("trace_p2p", None) # Remove trace from main results
            
            all_results.append({
                "experiment_name": exp_name,
                "seed": seed,
                "delta": delta,
                "profit": res["profit"],
                "p2p_volume": res["p2p_volume"],
                "grid_import": res["grid_import"],
                "grid_export": res["grid_export"]
            })
            
    # Phase 3: Compilation and Statistics
    df = pd.DataFrame(all_results)
    trace_df = pd.DataFrame(all_traces)
    
    os.makedirs("research_q1/results", exist_ok=True)
    out_file = "research_q1/results/results_all_experiments.csv"
    df.to_csv(out_file, index=False)
    trace_file = "research_q1/results/results_traces.csv"
    trace_df.to_csv(trace_file, index=False)
    print(f"\nAll experimental permutations completed. Raw data saved to {out_file} and {trace_file}.")
    
    # Calculate group statistics
    stats_df = df.groupby(["experiment_name", "delta"]).agg(
        profit_mean=("profit", "mean"),
        profit_std=("profit", "std"),
        p2p_mean=("p2p_volume", "mean"),
        p2p_std=("p2p_volume", "std"),
        import_mean=("grid_import", "mean"),
        import_std=("grid_import", "std")
    ).reset_index()
    
    print("\n--- Summary Statistics ---")
    print(stats_df.to_string())
    
    # Phase 4: Generate LaTeX Table
    print("\n--- LaTeX Table Generation ---")
    latex_str = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_str += "Model & Profit (\\$) & P2P Volume (kWh) & Grid Import (kWh) \\\\\n\\hline\n"
    
    for _, row in stats_df.iterrows():
        # Only include core experiments in the main table
        if "delta_sweep" not in row['experiment_name'] or row['delta'] == 0.03:
            name = row['experiment_name'].replace("_", "\_")
            latex_str += f"{name} & {row['profit_mean']:.2f} $\pm$ {row['profit_std']:.2f} & "
            latex_str += f"{row['p2p_mean']:.2f} $\pm$ {row['p2p_std']:.2f} & "
            latex_str += f"{row['import_mean']:.2f} $\pm$ {row['import_std']:.2f} \\\\\n"
            
    latex_str += "\\hline\n\\end{tabular}\n\\caption{Comparative Performance of Market Formulations}\n\\label{tab:results}\n\\end{table}"
    
    print(latex_str)
    
if __name__ == "__main__":
    run_all_experiments()
