
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv
from baselines.rule_based_agent import RuleBasedAgent

# CONSTANTS
DATA_FILE = "processed_hybrid_data.csv"
PEAK_HOURS = range(17, 21) 
N_EPISODES = 10 

def run_one_env(env, agents, n):
    metrics = []
    obs, _ = env.reset()
    for _ in range(24):
        if agents:
            obs_reshaped = obs.reshape(n, -1)
            action = np.array([agents[i].get_action(obs_reshaped[i], (env.current_idx % 24)) for i in range(n)]).flatten()
        else:
            action = np.zeros((n, 3)).flatten()
        obs, reward, done, truncated, info = env.step(action)
        
        # Calculate carbon (Grid Import * Intensity)
        grid_imp = info.get('total_import', 0)
        intensity = info.get('carbon_intensity', 0.4)
        carbon = grid_imp * intensity
        
        metrics.append({
            'hour': (env.current_idx - 1) % 24,
            'p2p_volume': info.get('p2p_volume', 0),
            'grid_import': grid_imp,
            'carbon_kg': carbon,
            'reward': reward,
            'violations': info.get('lagrangian/violation_soc', 0) + info.get('lagrangian/violation_line', 0) + info.get('lagrangian/violation_voltage', 0)
        })
        if done or truncated: break
    return pd.DataFrame(metrics)

def run_rl_model_robust(path, n, seed, norm=None):
    if not os.path.exists(path + ".zip"): return None
    try:
        model = PPO.load(path)
        eval_env = VectorizedMultiAgentEnv(n_agents=n)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        if norm and os.path.exists(norm):
            eval_env = VecNormalize.load(norm, eval_env)

        else:
            eval_env.reset()
            # quick stabilize
            for _ in range(100): 
                actions = np.array([eval_env.action_space.sample() for _ in range(n)])
                eval_env.step(actions)
        
        eval_env.env_method('update_training_step', 1000000)
        eval_env.seed(seed)
        obs = eval_env.reset()
        metrics = []
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            info = infos[0]
            
            # Calculate carbon
            grid_imp = info.get('total_import', 0)
            intensity = info.get('carbon_intensity', 0.4)
            carbon = grid_imp * intensity
            
            metrics.append({
                'hour': (eval_env.env.current_idx - 1) % 24,
                'p2p_volume': info.get('p2p_volume', 0),
                'grid_import': grid_imp,
                'carbon_kg': carbon,
                'reward': np.mean(rewards),
                'violations': info.get('lagrangian/violation_soc', 0) + info.get('lagrangian/violation_line', 0) + info.get('lagrangian/violation_voltage', 0)
            })
        eval_env.close()
        return pd.DataFrame(metrics)
    except Exception as e:
        print(f"  Error loading model {path}: {e}")
        return None

def evaluate_all_configs(n, include_v21=False):
    all_metrics = {
        'Grid-Only': [],
        'Legacy Auction': [],
        'SLIM v5': []
    }
    if include_v21: all_metrics['SLIM v2.1'] = []
    
    for ep in range(N_EPISODES):
        seed = 42 + ep
        print(f"  Episode {ep+1}/{N_EPISODES} (Seed {seed})")
        
        # Grid Only
        env_g = EnergyMarketEnvRobust(n_prosumers=int(n*0.6), n_consumers=n-int(n*0.6), data_file=DATA_FILE, random_start_day=True)
        env_g.reset(seed=seed)
        df_g = run_one_env(env_g, None, n)
        all_metrics['Grid-Only'].append(df_g)
        
        # Auction
        env_a = EnergyMarketEnvRobust(n_prosumers=int(n*0.6), n_consumers=n-int(n*0.6), data_file=DATA_FILE, random_start_day=True)
        env_a.use_alignment_reward = False
        env_a.use_curriculum = False
        env_a.reset(seed=seed)
        agents = [RuleBasedAgent(i, env_a.nodes[i].battery_capacity_kwh, env_a.nodes[i].battery_max_charge_kw) for i in range(n)]
        df_a = run_one_env(env_a, agents, n)
        all_metrics['Legacy Auction'].append(df_a)
        
        # SLIM v5
        model_v5_path = f"models_scalable_v5/ppo_n{n}_v5"
        df_v5 = run_rl_model_robust(model_v5_path, n, seed)
        if df_v5 is not None: all_metrics['SLIM v5'].append(df_v5)
        
        # SLIM v2.1 (only for n=4)
        if include_v21:
            df_v21 = run_rl_model_robust("models_slim/seed_0/best_model", 4, seed, "models_slim/seed_0/vec_normalize.pkl")
            if df_v21 is not None: all_metrics['SLIM v2.1'].append(df_v21)
            
    # Aggregate
    agg_results = []
    for name, dfs in all_metrics.items():
        if not dfs: continue
        combined = pd.concat(dfs)
        
        # Windows: Peak and Full Day
        for window_name, hours in [("Peak (17-21)", PEAK_HOURS), ("Full Day (24h)", range(24))]:
            window_df = combined[combined['hour'].isin(hours)]
            
            agg_results.append({
                'Model': name,
                'Window': window_name,
                'P2P Volume (kWh)': window_df['p2p_volume'].sum() / N_EPISODES,
                'Grid Import (kWh)': window_df['grid_import'].sum() / N_EPISODES,
                'CO2 Emission (kg)': window_df['carbon_kg'].sum() / N_EPISODES,
                'Avg Reward': window_df['reward'].mean(),
                'Safety Violations': int(window_df['violations'].sum() / N_EPISODES)
            })
    return agg_results

def to_md(df):
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"]*len(df.columns)) + " |"
    rows = []
    for row in df.values:
        rows.append("| " + " | ".join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]) + " |")
    return "\n".join([header, sep] + rows)

def main():
    print(f"\nEvaluating over {N_EPISODES} random episodes of 24h each...")
    
    # Run evaluations
    print("\n--- PHASE 1: N=4 ---")
    res_n4 = evaluate_all_configs(4, include_v21=True)
    
    print("\n--- PHASE 2: N=24 ---")
    res_n24 = evaluate_all_configs(24, include_v21=False)
    
    df4 = pd.DataFrame(res_n4)
    df4['Scale'] = 'N=4'
    df24 = pd.DataFrame(res_n24)
    df24['Scale'] = 'N=24'
    
    combined_df = pd.concat([df4, df24])
    combined_df.to_csv("research_q1/results/scientific_proof_data.csv", index=False)
    print("\nRESULTS GENERATED AND SAVED TO CSV.")
    
    # Legacy MD Report
    os.makedirs("research_q1/results", exist_ok=True)
    report_path = "research_q1/results/slim_v5_comparison.md"
    with open(report_path, "w") as f:
        f.write("# SLIM v5 vs Baselines: Performance Benchmark\n\n")
        f.write(f"Aggregated over {N_EPISODES} random days.\n\n")
        f.write("## 1. Small Scale (N=4)\n")
        f.write(to_md(df4))
        f.write("\n\n## 2. Large Scale (N=24)\n")
        f.write(to_md(df24))
    
    print(f"Markdown report saved to {report_path}")

if __name__ == "__main__":
    main()
