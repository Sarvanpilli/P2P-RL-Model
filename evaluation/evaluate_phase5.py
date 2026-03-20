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
            
            if t == 0:
                print(f"\nDEBUG STEP 0 ({mode}):")
                print(f"  Actions: {action}")
                print(f"  Info Keys: {list(info.keys())}")
                print(f"  P2P Volume Step: {info.get('p2p_volume_kwh_step', 'MISSING')}")
                print(f"  Total Import: {info.get('total_import', 'MISSING')}")
                print(f"  Reward: {reward}")
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
            step_metrics["p2p_volume"] = info.get("p2p_volume_kwh_step", 0.0)
            
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
            
            # Phase 5 Observation is complex. For Rule-Based baseline, we'll
            # extract the raw physical values directly from the environment
            # to ensure the heuristic works as intended.
            demands, pvs, co2 = env_base._get_current_data()
            socs = np.array([n.soc for n in env_base.nodes])
            retail, feed_in = env_base._get_grid_prices()
            
            for i in range(env_base.n_agents):
                # Construct a 'physical' observation for the RuleBasedAgent [Dem, SoC, PV, ..., Retail, FeedIn]
                physical_obs = np.zeros(8)
                physical_obs[0] = demands[i]
                physical_obs[1] = socs[i]
                physical_obs[2] = pvs[i]
                physical_obs[6] = retail
                physical_obs[7] = feed_in
                
                act = agents[i].get_action(physical_obs, t % 24)
                actions.append(act)
            
            flat_action = np.array(actions).flatten()
            obs, reward, done, trunc, info = env_base.step(flat_action)
            
            # DEBUG: Print first few steps
            if t < 5:
                print(f"  Step {t} Baseline: Demand={demands[0]:.2f}, PV={pvs[0]:.2f}, Action={actions[0]}, P2P={info.get('p2p_volume_kwh_step', 0.0):.2f}")
            
            step_metrics["market_price"] = info.get("market_price", 0.0)
            step_metrics["loss_kw"] = info.get("loss_kw", 0.0)
            step_metrics["total_import"] = info.get("total_import", 0.0)
            step_metrics["total_export"] = info.get("total_export", 0.0)
            step_metrics["net_grid_flow"] = info.get("total_export", 0.0) - info.get("total_import", 0.0)
            step_metrics["total_reward"] = np.sum(reward)
            step_metrics["soc_mean"] = np.mean([n.soc for n in env_base.nodes])
            step_metrics["soc_std"] = np.std([n.soc for n in env_base.nodes])
            step_metrics["p2p_volume"] = info.get("p2p_volume_kwh_step", 0.0)
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

def run_multiseed_evaluation(model_paths, env_config, n_eval_steps=336):
    """
    Run evaluation across multiple seeds and compute stats.
    Args:
        model_paths: List of paths to .zip models
        env_config: Dict with env parameters
        n_eval_steps: Timesteps per seed
    """
    seed_results = []
    metrics_to_track = ['p2p_volume', 'total_import', 'total_reward', 'soc_mean', 'market_price']
    
    print(f"\nStarting Multiseed Evaluation ({len(model_paths)} seeds)...")
    
    for i, path in enumerate(model_paths):
        print(f"  Evaluating Seed {i} from: {path}")
        
        # Load Env
        def make_env():
            return EnergyMarketEnvRobust(**env_config)
            
        env_base = make_env()
        env = DummyVecEnv([lambda: env_base])
        
        # Load VecNormalize if exists
        vec_path = os.path.join(os.path.dirname(path), "vec_normalize.pkl")
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False
        
        # Load Model
        model = PPO.load(path, env=env)
        
        # Eval Loop (simplified for speed/stats)
        obs = env.reset()
        episode_metrics = {m: [] for m in metrics_to_track}
        
        for t in range(n_eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            info = infos[0]
            
            if i == 0 and t < 5:
                print(f"  [DEBUG Seed 0 Step {t}] Action={action[0]}, P2P={info.get('p2p_volume_kwh_step', 0.0):.4f}, Import={info.get('total_import', 0.0):.4f}")
            
            episode_metrics['p2p_volume'].append(info.get('p2p_volume_kwh_step', 0.0))
            episode_metrics['total_import'].append(info.get('total_import', 0.0))
            episode_metrics['total_reward'].append(np.sum(reward))
            episode_metrics['soc_mean'].append(np.mean([n.soc for n in env.envs[0].nodes]))
            episode_metrics['market_price'].append(info.get('market_price', 0.0))
            
        # Compute Episode Totals/Means for this seed
        seed_summary = {
            'p2p_volume':   np.sum(episode_metrics['p2p_volume']),
            'total_import':  np.sum(episode_metrics['total_import']),
            'total_reward': np.mean(episode_metrics['total_reward']), # mean per step
            'soc_mean':     np.mean(episode_metrics['soc_mean']),
            'market_price': np.mean(episode_metrics['market_price'])
        }
        seed_results.append(seed_summary)
        print(f"    Seed {i} Result: Reward={seed_summary['total_reward']:.2f}, P2P={seed_summary['p2p_volume']:.2f}")

    # Compute Aggregate Stats
    final_stats = {}
    n_seeds = len(seed_results)
    
    for metric in seed_results[0].keys():
        values = [res[metric] for res in seed_results]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n_seeds > 1 else 0.0
        ci_95 = 1.96 * (std / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        
        final_stats[metric] = {
            'mean': float(mean),
            'std': float(std),
            'ci_low': float(mean - ci_95),
            'ci_high': float(mean + ci_95)
        }

    # Print Formatted Table
    print("\n" + "┌" + "─"*21 + "┬" + "─"*14 + "┬" + "─"*14 + "┬" + "─"*14 + "┐")
    print("│ Metric              │ Mean         │ Std Dev      │ 95% CI       │")
    print("├" + "─"*21 + "┼" + "─"*14 + "┼" + "─"*14 + "┼" + "─"*14 + "┤")
    
    labels = {
        'p2p_volume': 'P2P Volume (kWh)',
        'total_import': 'Grid Import (kWh)',
        'total_reward': 'Total Reward',
        'soc_mean': 'Mean SoC (%)',
        'market_price': 'Market Price ($/kWh)'
    }
    
    for m, label in labels.items():
        s = final_stats[m]
        print(f"│ {label:<20}│ {s['mean']:<13.2f}│ ± {s['std']:<11.2f}│ [{s['ci_low']:<5.2f}, {s['ci_high']:<5.2f}] │")
        
    print("└" + "─"*21 + "┴" + "─"*14 + "┴" + "─"*14 + "┴" + "─"*14 + "┘")
    print(f"95% CI = mean ± 1.96 * (std / sqrt({n_seeds}))")
    
    return final_stats

def compare_models_multiseed(baseline_paths, auction_paths, slim_paths, env_config):
    """Compare three model architectures across seeds."""
    from datetime import datetime
    
    all_results = {}
    models = {
        "baseline": baseline_paths,
        "legacy_auction": auction_paths,
        "slim": slim_paths
    }
    
    for name, paths in models.items():
        if paths and len(paths) > 0:
            print(f"\nEvaluating Model Group: {name}")
            all_results[name] = run_multiseed_evaluation(paths, env_config)
    
    # Print Comparison Table
    print("\n" + "┌" + "─"*21 + "┬" + "─"*18 + "┬" + "─"*18 + "┬" + "─"*18 + "┐")
    print("│ Metric              │ Baseline         │ Legacy Auction   │ SLIM (Ours)      │")
    print("├" + "─"*21 + "┼" + "─"*18 + "┼" + "─"*18 + "┼" + "─"*18 + "┤")
    
    metrics = ['p2p_volume', 'total_import']
    labels = {'p2p_volume': 'P2P Volume (kWh)', 'total_import': 'Grid Import (kWh)'}
    
    for m in metrics:
        row = f"│ {labels[m]:<20}│"
        for name in ["baseline", "legacy_auction", "slim"]:
            if name in all_results:
                s = all_results[name][m]
                row += f" {s['mean']:>7.2f} ± {s['std']:<7.2f}│"
            else:
                row += f" {'N/A':<17}│"
        print(row)
        
    print("└" + "─"*21 + "┴" + "─"*18 + "┴" + "─"*18 + "┴" + "─"*18 + "┘")
    
    # Save to JSON
    import json
    save_data = {
        **all_results,
        "n_seeds": len(slim_paths) if slim_paths else 0,
        "timestamp": datetime.now().isoformat()
    }
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/multiseed_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nResults saved to evaluation/results/multiseed_results.json")


def main():
    """Run comprehensive evaluation including multiseed analysis"""
    
    # 1. Individual Checkpoint Evaluation (Original Logic)
    phase5_dir = "models_phase5_hybrid"
    checkpoints = [
        ("250k", "ppo_hybrid_250016_steps.zip"),
    ]
    
    for name, checkpoint in checkpoints:
        model_path = os.path.join(phase5_dir, checkpoint)
        if os.path.exists(model_path):
            run_evaluation(mode=f"RL_Phase5_{name}", model_path=model_path, 
                           output_csv=f"evaluation/results_phase5_{name}.csv", n_steps=168)
    
    # 2. PROMPT 3: MULTISEED EVALUATION
    print(f"\n{'#'*60}")
    print(f"# RUNNING MULTISEED COMPARISON (PROMPT 3)")
    print(f"{'#'*60}")
    
    env_config = {
        "n_agents": 4,
        "data_file": "evaluation/ausgrid_p2p_energy_dataset.csv",
        "random_start_day": False,
        "enable_ramp_rates": True,
        "enable_losses": True,
        "forecast_horizon": 4,
        "enable_predictive_obs": True,
        "diversity_mode": True
    }
    
    slim_seeds = [f"models_slim/seed_{i}/best_model.zip" for i in range(5)]
    slim_paths = [p for p in slim_seeds if os.path.exists(p)]
    
    if len(slim_paths) > 0:
        # Run multiseed comparison
        # (We use empty lists for baseline/legacy if not available to at least get SLIM stats)
        compare_models_multiseed(
            baseline_paths=[], # Rule-based is handled separately or can be added
            auction_paths=[], 
            slim_paths=slim_paths,
            env_config=env_config
        )
    else:
        print("ERROR: No multiseed checkpoints found in models_slim/")

if __name__ == "__main__":
    main()
