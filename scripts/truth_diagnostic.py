
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

DATA_FILE = "processed_hybrid_data.csv"
PEAK_HOURS = range(17, 21)
N_AGENTS = 24
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
        metrics.append({
            'hour': (env.current_idx - 1) % 24,
            'p2p_volume': info.get('p2p_volume', 0),
            'grid_import': info.get('total_import', 0),
            'reward': reward,
            'violations': info.get('lagrangian/violation_soc', 0) + info.get('lagrangian/violation_line', 0) + info.get('lagrangian/violation_voltage', 0)
        })
        if done or truncated: break
    return pd.DataFrame(metrics)

def run_rl_model_robust(path, n, seed, norm_path=None):
    if not os.path.exists(path + ".zip"): return None
    try:
        model = PPO.load(path)
        eval_env = VectorizedMultiAgentEnv(n_agents=n)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        if norm_path and os.path.exists(norm_path):
            eval_env = VecNormalize.load(norm_path, eval_env)
        else:
            eval_env.reset()
            for _ in range(50):
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
            metrics.append({
                'hour': (eval_env.env.current_idx - 1) % 24,
                'p2p_volume': info.get('p2p_volume', 0),
                'grid_import': info.get('total_import', 0),
                'reward': np.mean(rewards),
                'violations': info.get('lagrangian/violation_soc', 0) + info.get('lagrangian/violation_line', 0) + info.get('lagrangian/violation_voltage', 0)
            })
        eval_env.close()
        return pd.DataFrame(metrics)
    except: return None

def run_truth_bench():
    print(f"Starting Truth Bench for N={N_AGENTS} over {N_EPISODES} episodes...")
    model_path = f"models_scalable_v5/ppo_n{N_AGENTS}_v5"
    results = []
    seeds = [100 + i for i in range(N_EPISODES)]

    for ep, seed in enumerate(seeds):
        print(f"  Episode {ep+1}/{N_EPISODES} (Seed {seed})")
        
        # 1. Grid Only
        env_g = EnergyMarketEnvRobust(n_prosumers=int(N_AGENTS*0.6), n_consumers=N_AGENTS-int(N_AGENTS*0.6), data_file=DATA_FILE, random_start_day=True)
        env_g.reset(seed=seed)
        df_g = run_one_env(env_g, None, N_AGENTS)
        
        # 2. Legacy Auction
        env_a = EnergyMarketEnvRobust(n_prosumers=int(N_AGENTS*0.6), n_consumers=N_AGENTS-int(N_AGENTS*0.6), data_file=DATA_FILE, random_start_day=True)
        env_a.use_alignment_reward = False
        env_a.use_curriculum = False
        env_a.reset(seed=seed)
        agents = [RuleBasedAgent(i, env_a.nodes[i].battery_capacity_kwh, env_a.nodes[i].battery_max_charge_kw) for i in range(N_AGENTS)]
        df_a = run_one_env(env_a, agents, N_AGENTS)
        
        # 3. SLIM v5
        df_v5 = run_rl_model_robust(model_path, N_AGENTS, seed)
        
        for name, df in [('Grid-Only', df_g), ('Legacy Auction', df_a), ('SLIM v5', df_v5)]:
            if df is None: continue
            peak_df = df[df['hour'].isin(PEAK_HOURS)]
            results.append({
                'ep': ep,
                'seed': seed,
                'metric': name,
                'p2p': peak_df['p2p_volume'].sum(),
                'import': peak_df['grid_import'].sum(),
                'breach': peak_df['violations'].sum(),
                'reward': peak_df['reward'].mean()
            })

    final_df = pd.DataFrame(results)
    print("\n--- COMPARATIVE TRUTH (N=24, PEAK HOURS) ---")
    summary = final_df.groupby('metric')[['p2p', 'import', 'breach', 'reward']].agg(['mean', 'std'])
    print(summary.to_string())
    
    os.makedirs("research_q1/results", exist_ok=True)
    final_df.to_csv("research_q1/results/truth_head_to_head.csv", index=False)
    print(f"\nRaw results saved to research_q1/results/truth_head_to_head.csv")

if __name__ == "__main__":
    run_truth_bench()
