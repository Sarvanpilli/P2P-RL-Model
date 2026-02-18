"""
Recovery Model Evaluation Script

Evaluates the Phase 5 Recovery model and generates detailed timestep data
for comparison with the failed Phase 5 baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
except ImportError:
    from stable_baselines3 import PPO
    USING_RECURRENT = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_recovery import EnergyMarketEnvRecovery


def evaluate_recovery_model(model_path, vec_normalize_path, n_episodes=1, episode_length=168):
    """
    Evaluate recovery model and collect detailed timestep data.
    
    Args:
        model_path: Path to trained model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to run
        episode_length: Length of each episode (168 = 1 week)
    
    Returns:
        DataFrame with timestep-level results
    """
    
    print(f"\n{'='*60}")
    print("RECOVERY MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Episode Length: {episode_length} hours")
    print(f"{'='*60}\n")
    
    # Create environment
    def make_env():
        env = EnergyMarketEnvRecovery(
            n_agents=4,
            data_file="processed_hybrid_data.csv",
            random_start_day=False,  # Start from beginning for consistency
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=4,
            seed=42
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    if os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"⚠ VecNormalize stats not found: {vec_normalize_path}")
        print("Proceeding without normalization...")
    
    # Load model
    print(f"Loading model from: {model_path}")
    if USING_RECURRENT:
        model = RecurrentPPO.load(model_path, env=env)
    else:
        model = PPO.load(model_path, env=env)
    
    print("✓ Model loaded successfully!\n")
    
    # Collect data
    all_data = []
    
    for episode in range(n_episodes):
        print(f"Running Episode {episode + 1}/{n_episodes}...")
        
        obs = env.reset()
        if USING_RECURRENT:
            lstm_states = None
        
        episode_reward = 0
        
        for step in range(episode_length):
            # Get action from model
            if USING_RECURRENT:
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Extract info from the underlying environment
            env_info = env.envs[0].unwrapped
            
            # Get per-agent data
            for agent_id in range(4):
                node = env_info.nodes[agent_id]
                
                # Determine agent type
                agent_types = ['Solar', 'Wind', 'EV', 'Standard']
                agent_type = agent_types[node.agent_type_id]
                
                # Get P2P trade for this agent (from last step)
                # We need to track this from the info dict
                p2p_trade = 0.0  # Will be populated from market results
                
                # Get grid flows
                grid_import = 0.0
                grid_export = 0.0
                
                # Record data
                data_row = {
                    'episode': episode,
                    'hour': step,
                    'agent_id': agent_id,
                    'agent_type': agent_type,
                    'soc': node.soc,
                    'soc_pct': node.soc / node.battery_capacity_kwh if node.battery_capacity_kwh > 0 else 0,
                    'capacity': node.battery_capacity_kwh,
                    'p2p_trade': p2p_trade,  # Placeholder
                    'grid_import': grid_import,  # Placeholder
                    'grid_export': grid_export,  # Placeholder
                    'reward': reward[0] / 4,  # Approximate per-agent reward
                    'p2p_volume': info[0].get('p2p_volume', 0),
                    'market_price': info[0].get('market_price', 0),
                    'total_export': info[0].get('total_export', 0),
                    'total_import': info[0].get('total_import', 0),
                    'mean_soc_pct': info[0].get('mean_soc_pct', 0)
                }
                
                all_data.append(data_row)
            
            if done[0]:
                break
        
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Mean SoC: {info[0].get('mean_soc_pct', 0):.2%}")
        print(f"  Total P2P Volume: {sum([d['p2p_volume'] for d in all_data[-episode_length*4:]]):.2f} kW")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total timesteps: {len(df)}")
    print(f"Mean SoC: {df['soc_pct'].mean():.2%}")
    print(f"Min SoC: {df['soc_pct'].min():.2%}")
    print(f"Max SoC: {df['soc_pct'].max():.2%}")
    print(f"Total P2P Volume: {df['p2p_volume'].sum():.2f} kW")
    print(f"Mean Reward: {df['reward'].mean():.4f}")
    print(f"{'='*60}\n")
    
    return df


def main():
    # Configuration
    MODEL_PATH = "models_phase5_recovery/ppo_recovery_single_51200_steps.zip"
    VEC_NORMALIZE_PATH = "models_phase5_recovery/vec_normalize_single.pkl"
    OUTPUT_PATH = "evaluation/results_recovery.csv"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("\nAvailable models:")
        if os.path.exists("models_phase5_recovery"):
            for f in os.listdir("models_phase5_recovery"):
                if f.endswith(".zip"):
                    print(f"  - {f}")
        return
    
    # Run evaluation
    results_df = evaluate_recovery_model(
        model_path=MODEL_PATH,
        vec_normalize_path=VEC_NORMALIZE_PATH,
        n_episodes=1,
        episode_length=168
    )
    
    # Save results
    os.makedirs("evaluation", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Results saved to: {OUTPUT_PATH}")
    
    # Calculate key metrics
    print(f"\n{'='*60}")
    print("KEY METRICS")
    print(f"{'='*60}")
    
    # Self-Sufficiency Rate (approximation)
    total_p2p = results_df['p2p_volume'].sum()
    total_grid_import = results_df['total_import'].sum()
    total_demand_proxy = total_p2p + total_grid_import
    
    if total_demand_proxy > 0:
        self_sufficiency = (total_p2p / total_demand_proxy) * 100
        print(f"Self-Sufficiency Rate: {self_sufficiency:.2f}%")
        print(f"  (P2P / (P2P + Grid Imports))")
    
    # P2P Liquidity Factor
    if total_grid_import > 0:
        liquidity_factor = total_p2p / total_grid_import
        print(f"P2P Liquidity Factor: {liquidity_factor:.4f}")
        print(f"  (Total P2P / Total Grid Imports)")
    else:
        print(f"P2P Liquidity Factor: ∞ (No grid imports!)")
    
    print(f"\nP2P Trading Activity:")
    print(f"  Total P2P Volume: {total_p2p:.2f} kW")
    print(f"  Total Grid Import: {total_grid_import:.2f} kW")
    print(f"  Total Grid Export: {results_df['total_export'].sum():.2f} kW")
    
    print(f"\nBattery Health:")
    print(f"  Mean SoC: {results_df['soc_pct'].mean():.2%}")
    print(f"  Min SoC: {results_df['soc_pct'].min():.2%}")
    print(f"  Max SoC: {results_df['soc_pct'].max():.2%}")
    print(f"  Std SoC: {results_df['soc_pct'].std():.2%}")
    
    print(f"\nReward Performance:")
    print(f"  Mean Reward: {results_df['reward'].mean():.4f}")
    print(f"  Total Reward: {results_df['reward'].sum():.2f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
