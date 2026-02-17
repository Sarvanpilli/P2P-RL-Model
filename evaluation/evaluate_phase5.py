"""
Phase 5 Hybrid Model Evaluation Script

Evaluates the Phase 5 Hybrid model checkpoint and compares against baselines.
Generates comprehensive metrics for performance analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
except ImportError:
    USING_RECURRENT = False
    print("RecurrentPPO not available, using standard PPO")

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
from baselines.rule_based_agent import RuleBasedAgent


def run_evaluation(mode="RL_Phase5", model_path=None, output_csv="eval_output.csv", n_steps=336):
    """
    Run evaluation for specified mode
    
    Args:
        mode: Evaluation mode (RL_Phase5, Baseline, etc.)
        model_path: Path to trained model
        output_csv: Output CSV file path
        n_steps: Number of timesteps to evaluate (default: 336 = 2 weeks)
    """
    print(f"\n{'='*60}")
    print(f"Running Evaluation: {mode}")
    print(f"{'='*60}\n")
    
    # Setup Environment - Using REAL DATA
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4,
            data_file="evaluation/ausgrid_p2p_energy_dataset.csv",
            random_start_day=False,
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            enable_predictive_obs=True,
            forecast_noise_std=0.05,
            diversity_mode=True
        )
    
    env_base = make_env()
    
    # Load Model for RL modes
    model = None
    env = None
    
    if "RL" in mode:
        # Wrap for VecNormalize
        env = DummyVecEnv([lambda: env_base])
        
        # Load VecNormalize stats
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(vec_path):
            print(f"Loading VecNormalize stats from: {vec_path}")
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False
        else:
            print("WARNING: No VecNormalize stats found!")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            env.training = False
            env.norm_reward = False
        
        # Load model
        try:
            print(f"Loading model from: {model_path}")
            if USING_RECURRENT:
                try:
                    model = RecurrentPPO.load(model_path, env=env)
                    print("Loaded RecurrentPPO model")
                except Exception as e:
                    print(f"RecurrentPPO load failed: {e}, trying PPO...")
                    model = PPO.load(model_path, env=env)
            else:
                model = PPO.load(model_path, env=env)
            
            print(f"Model loaded successfully!")
            print(f"Model timesteps: {model.num_timesteps:,}")
            
        except Exception as e:
            print(f"ERROR: Model load failed: {e}")
            return None
        
        # Initialize LSTM states if using RecurrentPPO
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        # Reset environment
        obs = env.reset()
        
    else:
        # Baseline Agents
        agents = [RuleBasedAgent(i, 50.0, 25.0) for i in range(env_base.n_agents)]
        obs = env_base.reset()[0]
    
    # Evaluation loop
    results = []
    episode_rewards = []
    current_episode_reward = 0
    
    print(f"\nStarting evaluation for {n_steps} timesteps...")
    
    for t in range(n_steps):
        step_metrics = {
            "step": t,
            "mode": mode
        }
        
        if "RL" in mode:
            # RL Agent prediction
            if USING_RECURRENT and isinstance(model, RecurrentPPO):
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=True
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            
            # Access underlying environment
            raw_env = env.envs[0]
            nodes = raw_env.nodes
            
            # Collect metrics
            step_metrics["market_price"] = info.get("market_price", 0.0)
            step_metrics["loss_kw"] = info.get("loss_kw", 0.0)
            step_metrics["total_import"] = info.get("total_import", 0.0)
            step_metrics["total_export"] = info.get("total_export", 0.0)
            step_metrics["net_grid_flow"] = info.get("total_export", 0.0) - info.get("total_import", 0.0)
            step_metrics["total_reward"] = np.sum(reward)
            step_metrics["soc_mean"] = np.mean([n.soc for n in nodes])
            step_metrics["soc_std"] = np.std([n.soc for n in nodes])
            step_metrics["p2p_volume"] = info.get("p2p_volume", 0.0)
            
            # Phase 5 specific metrics
            step_metrics["smoothing_penalty"] = info.get("reward/smoothing_mean", 0.0)
            step_metrics["grid_penalty"] = info.get("reward/grid_penalty_mean", 0.0)
            
            current_episode_reward += np.sum(reward)
            
            if dones[0]:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                episode_starts = np.ones((1,), dtype=bool)
                lstm_states = None
            
        else:
            # Baseline Loop
            actions = []
            obs_per_agent = len(obs) // env_base.n_agents
            
            for i in range(env_base.n_agents):
                agent_obs = obs[i*obs_per_agent : (i+1)*obs_per_agent]
                act = agents[i].get_action(agent_obs, t)
                actions.append(act)
            
            flat_action = np.array(actions).flatten()
            obs, reward, done, trunc, info = env_base.step(flat_action)
            
            step_metrics["market_price"] = info.get("market_price", 0.0)
            step_metrics["loss_kw"] = info.get("loss_kw", 0.0)
            step_metrics["total_import"] = info.get("total_import", 0.0)
            step_metrics["total_export"] = info.get("total_export", 0.0)
            step_metrics["net_grid_flow"] = info.get("total_export", 0.0) - info.get("total_import", 0.0)
            step_metrics["total_reward"] = np.sum(reward)
            step_metrics["soc_mean"] = np.mean([n.soc for n in env_base.nodes])
            step_metrics["soc_std"] = np.std([n.soc for n in env_base.nodes])
            step_metrics["p2p_volume"] = info.get("p2p_volume", 0.0)
            step_metrics["smoothing_penalty"] = 0.0
            step_metrics["grid_penalty"] = 0.0
            
            current_episode_reward += np.sum(reward)
        
        results.append(step_metrics)
        
        if (t + 1) % 24 == 0:
            print(f"  Progress: {t+1}/{n_steps} steps ({(t+1)/n_steps*100:.1f}%)")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"Total timesteps: {len(df)}")
    print(f"\nSummary Statistics:")
    print(f"  Mean Reward: {df['total_reward'].mean():.2f}")
    print(f"  Total Import: {df['total_import'].sum():.2f} kWh")
    print(f"  Total Export: {df['total_export'].sum():.2f} kWh")
    print(f"  Mean SoC: {df['soc_mean'].mean():.2f}%")
    print(f"  P2P Volume: {df['p2p_volume'].sum():.2f} kWh")
    print(f"{'='*60}\n")
    
    return df


def main():
    """Run comprehensive evaluation"""
    
    # Check for Phase 5 model checkpoints
    phase5_dir = "models_phase5_hybrid"
    
    # Find latest checkpoint
    checkpoints = [
        ("50k", "ppo_hybrid_50000_steps.zip"),
        ("100k", "ppo_hybrid_100000_steps.zip"),
        ("150k", "ppo_hybrid_150000_steps.zip"),
        ("200k", "ppo_hybrid_200000_steps.zip"),
        ("250k", "ppo_hybrid_250016_steps.zip"),
    ]
    
    results_summary = []
    
    # Evaluate each checkpoint
    for name, checkpoint in checkpoints:
        model_path = os.path.join(phase5_dir, checkpoint)
        if os.path.exists(model_path):
            print(f"\n{'#'*60}")
            print(f"# Evaluating Checkpoint: {name}")
            print(f"{'#'*60}")
            
            output_csv = f"evaluation/results_phase5_{name}.csv"
            df = run_evaluation(
                mode=f"RL_Phase5_{name}",
                model_path=model_path,
                output_csv=output_csv,
                n_steps=168  # 1 week
            )
            
            if df is not None:
                results_summary.append({
                    "checkpoint": name,
                    "mean_reward": df['total_reward'].mean(),
                    "total_import": df['total_import'].sum(),
                    "total_export": df['total_export'].sum(),
                    "mean_soc": df['soc_mean'].mean(),
                    "p2p_volume": df['p2p_volume'].sum()
                })
        else:
            print(f"Checkpoint not found: {model_path}")
    
    # Evaluate baseline
    print(f"\n{'#'*60}")
    print(f"# Evaluating Baseline")
    print(f"{'#'*60}")
    
    df_baseline = run_evaluation(
        mode="Baseline",
        model_path=None,
        output_csv="evaluation/results_phase5_baseline.csv",
        n_steps=168
    )
    
    if df_baseline is not None:
        results_summary.append({
            "checkpoint": "Baseline",
            "mean_reward": df_baseline['total_reward'].mean(),
            "total_import": df_baseline['total_import'].sum(),
            "total_export": df_baseline['total_export'].sum(),
            "mean_soc": df_baseline['soc_mean'].mean(),
            "p2p_volume": df_baseline['p2p_volume'].sum()
        })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("evaluation/phase5_evaluation_summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
