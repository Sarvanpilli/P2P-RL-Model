
import numpy as np
import pandas as pd
import gymnasium as gym
import os
import sys

# Parent Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_robust import EnergyMarketEnvRobust
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Scenario Wrappers ---

class CloudyEnv(EnergyMarketEnvRobust):
    """Scenario 1: Solar generation is 20% of expected."""
    def _get_current_data(self):
        dem, pv, co2 = super()._get_current_data()
        return dem, pv * 0.2, co2

class OutageEnv(EnergyMarketEnvRobust):
    """Scenario 2: Grid Price Spikes (Island Mode Pressure)."""
    def _get_grid_prices(self):
        # Grid is available but prohibitively expensive
        # Buy at $10.00/kWh, Sell at $0.00/kWh
        return 10.0, 0.0

class GreedyEnv(EnergyMarketEnvRobust):
    """Scenario 3: Neighbors refuse to trade unless price is max."""
    def step(self, action):
        # Manipulate actions of Agents 1, 2, 3 (Neighbors) to me "Greedy"
        # Action shape: (N, 3) -> [Charge, Trade, Bid]
        
        # We intercept the action before passing to super().step()
        # But wait, step() takes action from RL. 
        # In multi-agent training, 'action' is (N, 3).
        # We assume Agent 0 is the "Ego Agent" (RL under test).
        # Agents 1-3 act as Adversaries.
        
        # Create a copy so we don't mutate original buffer if shared
        modified_action = action.copy()
        
        # Neighbors (1, 2, 3) always Bid Max Price (1.0 -> $0.50)
        # And let's say they try to Sell (Trade > 0)
        # Or if they Buy, they Bid Min Price? Greedy usually means "Buy Low, Sell High".
        # Let's simple: Neighbors always Ask $0.50 when Selling.
        
        neighbor_indices = [1, 2, 3]
        for i in neighbor_indices:
            # If Selling (Action[1] > 0)
            if modified_action[i, 1] > 0:
                modified_action[i, 2] = 1.0 # Ask Max
            # If Buying (Action[1] < 0)
            elif modified_action[i, 1] < 0:
                modified_action[i, 2] = 0.0 # Bid Min (Greedy Buyer)
        
        return super().step(modified_action)

# --- Test Runner ---

def run_stress_test(model_path="models_phase5/ppo_robust.zip"):
    scenarios = {
        "Cloudy Week": CloudyEnv,
        "Grid Outage": OutageEnv,
        "Greedy Neighbors": GreedyEnv
    }
    
    # Check Model
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return

    results = []

    for name, EnvClass in scenarios.items():
        print(f"--- Running {name} ---")
        
        try:
            # Init Env
            # Note: We use Phase 5 Configs
            env = EnvClass(
                n_agents=4, 
                data_file="evaluation/ausgrid_p2p_energy_dataset.csv", # Real Data
                enable_predictive_obs=True,
                forecast_noise_std=0.05,
                diversity_mode=True
            )
            
            # Wrap
            vec_env = DummyVecEnv([lambda: env])
            
            # Load Model
            # Try Recurrent first
            try:
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(model_path, env=vec_env)
            except:
                model = PPO.load(model_path, env=vec_env)

            # Run Episode (1 week = 168 steps)
            obs = vec_env.reset()
            states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            total_reward = 0
            grid_imports = []
            
            for _ in range(168):
                action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)
                episode_starts = dones
                
                total_reward += np.sum(rewards)
                
                # Info extraction
                info = infos[0]
                grid_imports.append(info.get('total_import', 0))
            
            # Metrics
            avg_import = np.mean(grid_imports)
            print(f"[{name}] Reward: {total_reward:.2f}, Avg Import: {avg_import:.2f} kW")
            
            results.append({
                "Scenario": name,
                "Total Reward": total_reward,
                "Avg Grid Import (kW)": avg_import
            })
            
        except Exception as e:
            print(f"[{name}] Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save Report
    df = pd.DataFrame(results)
    print("\n--- Stress Test Report ---")
    print(df)
    df.to_csv("stress_test_results.csv", index=False)

if __name__ == "__main__":
    run_stress_test()
