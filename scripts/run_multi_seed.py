import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from train.energy_env_robust import EnergyMarketEnvRobust

def make_env(seed, rank):
    def _init():
        env = EnergyMarketEnvRobust(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=1,
            enable_predictive_obs=True,
            forecast_noise_std=0.05,
            diversity_mode=True
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def train_and_eval(seed: int, timesteps: int, results_dir: str):
    print(f"\n{'='*50}")
    print(f"Starting Training for Seed {seed}")
    print(f"{'='*50}")
    
    log_dir = os.path.join(results_dir, f"logs_seed_{seed}")
    model_dir = os.path.join(results_dir, f"models_seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    
    n_envs = 4
    env = SubprocVecEnv([make_env(seed, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    
    # Standard PPO (simpler base model for robust comparison)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=log_dir,
        seed=seed
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20480,
        save_path=model_dir,
        name_prefix=f"ppo_seed_{seed}"
    )
    
    model.learn(total_timesteps=timesteps, tb_log_name="run", callback=checkpoint_callback)
    
    model_path = os.path.join(model_dir, "final_model")
    model.save(model_path)
    env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    
    # Evaluation Phase
    print(f"\nEvaluating Seed {seed}...")
    eval_env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=False, # Deterministic start for fair eval
        forecast_horizon=1,
        enable_predictive_obs=True
    )
    
    # Wrap in dummy VecNormalize (inference mode)
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_venv = DummyVecEnv([lambda: eval_env])
    eval_venv = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), eval_venv)
    eval_venv.training = False
    eval_venv.norm_reward = False
    
    ep_rewards = []
    ep_profits = []
    ep_safe_violations = []
    ep_grid_import = []
    
    obs = eval_venv.reset()
    for _ in range(5): # Evaluate over 5 episodes
        done = False
        ep_ret = 0
        ep_prof = 0
        ep_viol = 0
        ep_imp = 0
        steps = 0
        
        while not done and steps < 168: # 1 week episodes
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = eval_venv.step(action)
            ep_ret += reward[0]
            
            info = info_list[0]
            ep_prof += info.get("mean_profit", 0)
            ep_viol += info.get("safety_violations", 0)
            ep_imp += info.get("total_import", 0)
            steps += 1
            
        ep_rewards.append(ep_ret)
        ep_profits.append(ep_prof)
        ep_safe_violations.append(ep_viol)
        ep_grid_import.append(ep_imp)
        
    metrics = {
        "Seed": seed,
        "Reward": np.mean(ep_rewards),
        "Profit": np.mean(ep_profits),
        "Safety_Violations": np.mean(ep_safe_violations),
        "Grid_Import_kW": np.mean(ep_grid_import)
    }
    
    print(f"Seed {seed} Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated random seeds")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps per seed")
    parser.add_argument("--out_dir", type=str, default="results_multiseed", help="Output directory")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_metrics = []
    for s in seeds:
        m = train_and_eval(s, args.timesteps, args.out_dir)
        all_metrics.append(m)
        
    # Compile Results
    df = pd.DataFrame(all_metrics)
    summary = df.agg(["mean", "std"]).T
    summary.columns = ["Mean", "StdDev"]
    
    print(f"\n{'='*50}")
    print("Aggregate Multi-Seed Results:")
    print(f"{'='*50}")
    print(summary)
    
    results_path = os.path.join(args.out_dir, "aggregate_metrics.csv")
    df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
