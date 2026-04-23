
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

DATA_FILE = "processed_hybrid_data.csv"
PEAK_HOURS = range(17, 21)

def run_diagnostic(n=24):
    model_path = f"models_scalable_v5/ppo_n{n}_v5"
    if not os.path.exists(model_path + ".zip"): return
    
    model = PPO.load(model_path)
    eval_env = VectorizedMultiAgentEnv(n_agents=n)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # Quick stabilize
    eval_env.reset()
    for _ in range(50): 
        # sample for all agents
        actions = np.array([eval_env.action_space.sample() for _ in range(n)])
        eval_env.step(actions)
    
    # Run one full day (24h) and log Agent 0
    obs = eval_env.reset()
    logs = []
    for t in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        
        # Extract features for Agent 0 from obs
        # obs is (24, 20). 
        # Feature 0: PV, 1: Demand, 2: SoC, 3: Net, ...
        a0_obs = obs[0]
        a0_info = infos[0] # info is usually community-wide or list?
        # In VectorizedMultiAgentEnv, info is a list of N dicts? 
        # Let's check. Actually eval_env.step returns infos tuple.
        
        hour = (eval_env.env.current_idx - 1) % 24
        logs.append({
            'hour': hour,
            'a0_pv': a0_obs[0],
            'a0_demand': a0_obs[1],
            'a0_soc': a0_obs[2],
            'a0_action_batt': action[0][0], # First discrete/cont? Cont in v5.
            'a0_grid': a0_info.get('total_import', 0) / n # Approximate per agent import
        })
        
    df = pd.DataFrame(logs)
    print("\n--- DIAGNOSTIC LOG (AGENT 0) ---")
    print(df[df['hour'].isin(PEAK_HOURS)].to_string(index=False))

if __name__ == "__main__":
    run_diagnostic(24)
