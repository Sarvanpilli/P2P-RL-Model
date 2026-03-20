import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

sns.set_theme(style="whitegrid", palette="muted")

def evaluate_scale_model(env, model, steps=500):
    obs = env.reset()
    p2p_volume = 0.0
    profit = 0.0
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        info = infos[0] 
        p2p_volume += info["p2p_volume"]
        profit = info["profit"] # Cumulative
        
    return profit, p2p_volume

def run_scalability_experiment():
    print("===========================================")
    print("STARTING SCALABILITY SWEEP (N=4, 6, 8, 10)")
    print("===========================================")
    
    scales = [4, 6, 8, 10]
    timesteps = 100_000
    eval_steps = 500
    seed = 42
    
    results = []
    
    for n in scales:
        print(f"\nTraining SLIM for N={n} Agents...")
        
        log_dir = f"research_q1/logs/scalability_test/N_{n}"
        os.makedirs(log_dir, exist_ok=True)
        
        def make_env():
            return EnergyMarketEnvSLIM(
                n_agents=n, 
                data_file="fixed_training_data_10.csv", 
                enable_safety=True,
                enable_p2p=True,
                market_type="dynamic",
                seed=seed
            )
            
        N_ENVS = 8
        env_fns = [make_env for _ in range(N_ENVS)]
        train_env = SubprocVecEnv(env_fns, start_method="spawn")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        model = PPO(
            "MlpPolicy", 
            train_env, 
            learning_rate=3e-4, 
            n_steps=1024, 
            batch_size=256, 
            n_epochs=10, 
            gamma=0.99, 
            gae_lambda=0.95, 
            tensorboard_log=log_dir,
            seed=seed,
            verbose=0
        )
        
        model.learn(total_timesteps=timesteps)
        
        # Save model
        model.save(os.path.join(log_dir, "ppo_model"))
        
        # Evaluation requires a fresh env or resetting normalize statistics
        # For simplicity in scalability testing, we'll use a DummyVecEnv for eval
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        # Sync normalization stats
        eval_env.obs_rms = train_env.obs_rms
        
        print(f"Evaluating N={n}...")
        prof, p2p = evaluate_scale_model(eval_env, model, steps=eval_steps)
        print(f"  -> Profit: ${prof:.2f} | P2P: {p2p:.2f} kWh")
        
        train_env.close()
        eval_env.close()
        
        # Normalize metrics per agent to allow fair comparison across scales
        results.append({
            "N": n,
            "Total Profit": prof,
            "Profit per Agent": prof / n,
            "Total P2P Volume": p2p,
            "P2P Volume per Agent": p2p / n
        })
        
    # Plot Scalability
    df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Participating Agents (N)', fontsize=12)
    ax1.set_ylabel('Avg Profit per Agent ($)', color=color, fontsize=12)
    sns.lineplot(data=df, x='N', y='Profit per Agent', marker='o', color=color, ax=ax1, linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Avg P2P Volume per Agent (kWh)', color=color, fontsize=12)
    sns.lineplot(data=df, x='N', y='P2P Volume per Agent', marker='s', color=color, ax=ax2, linewidth=3, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("SLIM Market Scalability Performance", fontsize=14, fontweight='bold')
    plt.xticks(scales)
    
    out_file = "research_q1/results/scalability_trend.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"\nSaved scalability plot to {out_file}")

if __name__ == "__main__":
    run_scalability_experiment()
