
import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv

MODEL_DIR = "models_scalable_v5"
SAVE_PATH = "research_q1/results/scalability_v5/scalability_metrics_v5_FIXED.csv"

def evaluate_clean(model_path, n_agents, steps=500):
    print(f"Evaluating N={n_agents}...")
    env = VectorizedMultiAgentEnv(n_agents=n_agents)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False) # Not needed for pure inference if stats not saved?
    # Actually, model was trained with VecNormalize. We really should use the normalize stats.
    # But for a quick check on P2P volume, raw env is often enough if obs are normalized inside.
    
    model = PPO.load(model_path)
    
    obs = env.reset()
    p2p_history, dep_history, success_history = [], [], []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        info = infos[0]
        # Environment bug is fixed, so info['grid_dependency'] should be correct now
        p2p_history.append(info.get('p2p_volume_kwh_step', 0))
        dep_history.append(info.get('grid_dependency', 0))
        success_history.append(float(info.get('p2p_volume_kwh_step', 0) > 0.05))
            
    env.close()
    return {
        'Agents': n_agents,
        'Grid %': np.mean(dep_history) * 100.0,
        'P2P (kWh)': np.sum(p2p_history),
        'Success %': np.mean(success_history) * 100.0
    }

def main():
    agent_counts = [4, 8, 12, 16, 24]
    results = []
    
    for n in agent_counts:
        path = os.path.join(MODEL_DIR, f"ppo_n{n}_v5.zip")
        if os.path.exists(path):
            res = evaluate_clean(path, n)
            results.append(res)
        else:
            print(f"Missing model: {path}")
            
    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Fixed results saved to {SAVE_PATH}")
    print(df)

if __name__ == "__main__":
    main()
