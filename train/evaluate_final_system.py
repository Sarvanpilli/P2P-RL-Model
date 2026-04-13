
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv

RESULTS_DIR = "research_q1/results/final_validation"
MODEL_DIR = "models_scalable_v5"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_config(name, n_agents=8, use_align=True, use_curr=True, steps=2000):
    print(f"\nEvaluating: {name} (N={n_agents})")
    model_path = os.path.join(MODEL_DIR, f"ppo_n{n_agents}_v5")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} not found.")
        return None
        
    from stable_baselines3.common.vec_env import VecNormalize
    
    model = PPO.load(model_path)
    eval_env = VectorizedMultiAgentEnv(
        n_agents=n_agents, 
        use_alignment_reward=use_align, 
        use_curriculum=use_curr
    )
    
    # Wrap in VecNormalize
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Progress set to 1.0 
    eval_env.env_method('update_training_step', 1000000) 
    
    # Warmup to stabilize normalization stats 
    print("Stabilizing normalization stats (1000 steps)...")
    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=False) # Use stochastic for better coverage
        obs, _, _, _ = eval_env.step(action)
    
    obs = eval_env.reset()
    metrics = {
        'success': [],
        'grid_dependency': [],
        'p2p_volume': [],
        'clean_profit': [],
        'economic_profit': [],
        'carbon': [],
        'demand': []
    }
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = eval_env.step(action)
        info = infos[0]
        
        metrics['success'].append(info.get('success', 0))
        dep = min(1.0, info.get('grid_dependency', 1.0))
        metrics['grid_dependency'].append(dep)
        metrics['p2p_volume'].append(info.get('p2p_volume', 0))
        metrics['clean_profit'].append(info.get('clean_profit_usd', 0))
        metrics['economic_profit'].append(info.get('economic_profit_usd', 0))
        metrics['carbon'].append(info.get('carbon_emissions_kg', 0))
        metrics['demand'].append(info.get('total_demand_kw', 0))
            
    eval_env.close()
    
    print(f"  Mean Step Success: {np.mean(metrics['success']):.2f}")
    print(f"  Mean Step Demand: {np.mean(metrics['demand']):.2f} kW")
    
    return {
        'Config': name,
        'Success %': np.mean(metrics['success']) * 100.0,
        'Grid %': np.mean(metrics['grid_dependency']) * 100.0,
        'P2P Vol': np.mean(metrics['p2p_volume']),
        'Profit ($)': np.sum(metrics['clean_profit']),
        'Economic ($)': np.sum(metrics['economic_profit']),
        'Carbon (kg)': np.sum(metrics['carbon'])
    }

def run_ablation_study():
    configs = [
        ("Full System", True, True),
        ("No Alignment", False, True),
        ("No Curriculum", True, False)
    ]
    results = []
    for name, align, curr in configs:
        res = evaluate_config(name, n_agents=8, use_align=align, use_curr=curr)
        if res: results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    
    # Plot Ablation
    plt.figure(figsize=(10, 6))
    plt.bar(df['Config'], df['Success %'], color=['green', 'red', 'orange'])
    plt.ylabel('Trade Success Rate (%)')
    plt.title('Ablation Study: Impact of Coordination Incentives (N=8)')
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_success.png"))
    plt.close()
    
    return df

def run_scalability_validation():
    counts = [4, 8, 12, 16, 24]   # PHASE 14: extended to N=24
    results = []
    for n in counts:
        res = evaluate_config(f"N={n}", n_agents=n, use_align=True, use_curr=True)
        if res: results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "scalability_validation.csv"), index=False)
    
    # Plot Scalability
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(counts, df['Success %'], marker='o', label='Success Rate', color='blue')
    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Success Rate (%)', color='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(counts, df['Grid %'], marker='s', label='Grid Dependency', color='red')
    ax2.set_ylabel('Grid Dependency (%)', color='red')
    
    plt.title('Scalability Validation: Performance vs. Agent Density')
    plt.savefig(os.path.join(RESULTS_DIR, "scalability_performance.png"))
    plt.close()
    
    # P2P Volume vs Agents
    plt.figure(figsize=(10, 6))
    plt.plot(counts, df['P2P Vol'], marker='^', color='green')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average P2P Volume (kWh/step)')
    plt.title('Market Liquidity Scaling')
    plt.savefig(os.path.join(RESULTS_DIR, "liquidity_scaling.png"))
    plt.close()
    
    return df

if __name__ == "__main__":
    print("Starting Final Scientific Validation...")
    ablation_df = run_ablation_study()
    print("\nAblation Results:")
    print(ablation_df.to_string(index=False))
    
    scalability_df = run_scalability_validation()
    print("\nScalability Results:")
    print(scalability_df.to_string(index=False))
    
    print(f"\nAll results and plots saved to {RESULTS_DIR}")
