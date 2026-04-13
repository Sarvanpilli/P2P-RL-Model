
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv

RESULTS_DIR = "research_q1/results/scalability_v5"
MODEL_DIR = "models_scalable_v5"
AGENT_COUNTS = [4, 8, 12, 16]

def evaluate_saved_model(n_agents, steps=2000):
    print(f"\n--- Final Evaluation N={n_agents} ---")
    model_path = os.path.join(MODEL_DIR, f"ppo_n{n_agents}_v5")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} not found. Skipping.")
        return None
        
    model = PPO.load(model_path)
    eval_env = VectorizedMultiAgentEnv(n_agents=n_agents)
    # Set progress to 1.0 for strict mode evaluation (no coordination rewards)
    eval_env.env_method('update_training_step', 1000000) 
    
    obs = eval_env.reset()
    p2p_vols = []
    grid_imports = []
    success_rates = []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = eval_env.step(action)
        info = infos[0]
        
        # Collect environment-level metrics
        p2p_vols.append(info.get('p2p_volume', 0))
        grid_imports.append(info.get('grid_import', 0))
        success_rates.append(info.get('trade_success_rate', 0))
            
    eval_env.close()
    
    avg_p2p = np.mean(p2p_vols)
    avg_grid = np.mean(grid_imports)
    # Grid Dependency = Grid Import / (Grid Import + P2P)
    grid_dep = (avg_grid / (avg_grid + avg_p2p + 1e-9)) * 100.0
    
    return {
        'Agents': n_agents,
        'Grid %': grid_dep,
        'P2P (kWh/step)': avg_p2p,
        'Success %': np.mean(success_rates) * 100.0
    }

def run_full_evaluation():
    summary = []
    for n in AGENT_COUNTS:
        res = evaluate_saved_model(n)
        if res: summary.append(res)
        
    df = pd.DataFrame(summary)
    print("\n" + "="*50 + "\nPHASE 12 FIXED EVALUATION RESULTS\n" + "="*50)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "fixed_scalability_metrics_v5.csv"), index=False)

if __name__ == "__main__":
    run_full_evaluation()
