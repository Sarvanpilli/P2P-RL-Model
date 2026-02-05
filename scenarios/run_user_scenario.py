
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
from stable_baselines3 import PPO

def run_simulation(n_prosumers=5, n_consumers=0, trained_only=False):
    n_agents = n_prosumers + n_consumers
    print(f"--- Starting User Scenario Simulation (Agents: {n_agents} [P:{n_prosumers}, C:{n_consumers}]) ---")
    
    # 0. Data Check
    # If we have many agents, we might need to regenerate data if current file is too small
    # For now, we rely on the env's fallback or existing file. 
    # But ideally we should auto-generate if N changed.
    import pandas as pd
    try:
        df = pd.read_csv('scenarios/user_scenario_data.csv')
        # Check if enough columns
        # Columns are agent_{i}_...
        # We need agent_{n_agents-1}
        needed_col = f"agent_{n_agents-1}_pv_kw"
        if needed_col not in df.columns:
            print(f"[Data] Existing data only has {len(df.columns)//2} agents. Needed {n_agents}.")
            print("[Data] Please run scenarios/generate_data.py with correct N first. Using Env fallback for missing columns.")
    except:
        pass

    # 1. Configuration
    battery_cap = 50.0 # kWh
    
    # 2. Setup Environment
    # We must match the training config: DummyVecEnv + VecNormalize
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=n_agents,
            n_prosumers=n_prosumers,
            n_consumers=n_consumers,
            timestep_hours=1.0,
            battery_capacity_kwh=battery_cap,
            data_file='scenarios/user_scenario_data.csv',
            random_start_day=False, # Force start at row 0
            forecast_horizon=1 # Matched to training
        )
    
    vec_env = DummyVecEnv([make_env])
    
    # Load Normalization Stats
    stats_path = "models/vec_normalize.pkl"
    if not os.path.exists(stats_path):
        stats_path = "train/models/vec_normalize.pkl"
        
    if os.path.exists(stats_path):
        print(f"[Init] Loading VecNormalize stats from {stats_path}")
        # We must load into the exact same env structure
        try:
            vec_env = VecNormalize.load(stats_path, vec_env)
            vec_env.training = False # Do not update stats during evaluation
            vec_env.norm_reward = False # We want raw rewards for logging
        except Exception as e:
            print(f"[Init] WARNING: VecNormalize load failed (Shape Mismatch?): {e}")
            if trained_only:
                 raise RuntimeError("VecNormalize load failed and --trained_only is set.")
    else:
        print("[Init] WARNING: VecNormalize stats not found. Model performance may be degraded.")
        if trained_only:
             print("[Init] CAUTION: Running without Normalization stats but --trained_only is set.")

    # 3. Initialize & Force State
    obs = vec_env.reset()
    real_env = vec_env.envs[0] # Unwrapped EnergyMarketEnvRobust
    
    # Force initial SoCs for first 5 agents if they exist (Legacy user request)
    initials_socs_pct = [0.65, 0.40, 0.30, 0.80, 0.50]
    print("\n[Init] Forcing Initial SoC State (First 5 Agents):")
    for i in range(min(n_agents, 5)):
        soc_pct = initials_socs_pct[i] if i < len(initials_socs_pct) else 0.5
        # If Consumer, SoC is 0 forced by class, but let's try setting it to check logic
        target_kwh = soc_pct * battery_cap
        real_env.nodes[i].reset(soc=target_kwh)
        actual = real_env.nodes[i].soc
        print(f"  Agent {i} ({real_env.nodes[i].agent_type.name}): Req {target_kwh:.2f} -> Set {actual:.2f} kWh")
        
    # 4. Load Agent
    model_path = "models/ppo_energy_final.zip" 
    if not os.path.exists(model_path):
        model_path = "train/models/ppo_energy_final.zip"
        
    model = None
    if os.path.exists(model_path):
        print(f"\n[Agent] Loading trained model from {model_path}")
        try:
            model = PPO.load(model_path)
            print("[Agent] Model loaded successfully.")
        except Exception as e:
            print(f"[Agent] ERROR: Failed to load model: {e}")
            model = None
    else:
        print("\n[Agent] Model file not found.")

    # STRICT ENFORCEMENT
    if trained_only and model is None:
        raise RuntimeError("ABORTING: --trained_only set but no valid PPO model found.")

    # 5. Run Episode
    history = {
        'soc': [],
        'market_price': [],
        'grid_export': [],
        'grid_import': [],
        'carbon': [],
        'rewards': []
    }
    
    print("\n[Sim] Running 24 Hour Episode...")
    print(f"{'Hour':<4} | {'Price ($)':<10} | {'Grid Net (kW)':<15} | {'Carbon (g)':<12}")
    print("-" * 50)
    
    for t in range(24):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Fallback (Only if trained_only=False)
            action = [vec_env.action_space.sample()]

        # Step
        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0]
        reward = rewards[0]
        
        # Log
        socs = [n.soc for n in real_env.nodes]
        history['soc'].append(socs)
        history['market_price'].append(info.get('market_price', 0))
        history['grid_export'].append(info.get('total_export', 0))
        history['grid_import'].append(info.get('total_import', 0))
        history['carbon'].append(info.get('total_carbon_mass', 0)) 
        history['rewards'].append(reward)
        
        net_grid = info.get('total_import',0) - info.get('total_export',0)
        carb = info.get('total_carbon_mass', 0)
        
        print(f"{t:<4} | {history['market_price'][-1]:<10.4f} | {net_grid:<15.2f} | {carb:<12.2f}")

    # 6. Plotting
    print("\n[Post] Generating Plots...")
    os.makedirs('results/user_scenario', exist_ok=True)
    
    # Plot 1: SoC Trajectories
    plt.figure(figsize=(10, 6))
    soc_data = np.array(history['soc']) / battery_cap * 100
    # Plot subset if too many
    plot_limit = min(n_agents, 10)
    for i in range(plot_limit):
        plt.plot(soc_data[:, i], label=f'Agent {i} ({real_env.nodes[i].agent_type.name[0]})')
    plt.title(f"Battery SoC Trajectories (First {plot_limit} Agents)")
    plt.xlabel("Hour")
    plt.ylabel("SoC %")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/user_scenario/soc_plot.png')
    plt.close()
    
    # Plot 2: Market Prices
    plt.figure(figsize=(10, 6))
    plt.plot(history['market_price'], color='green', marker='o')
    plt.title("P2P Market Clearing Price")
    plt.xlabel("Hour")
    plt.ylabel("Price ($/kWh)")
    plt.grid(True)
    plt.savefig('results/user_scenario/price_plot.png')
    plt.close()
    
    print("\nSuccess! Results saved to 'results/user_scenario/'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prosumers", type=int, default=5, help="Number of Prosumers")
    parser.add_argument("--n_consumers", type=int, default=0, help="Number of Consumers")
    parser.add_argument("--trained_only", action="store_true", help="Fail if trained model not found")
    args = parser.parse_args()
    
    run_simulation(
        n_prosumers=args.n_prosumers, 
        n_consumers=args.n_consumers, 
        trained_only=args.trained_only
    )

