# run_env_test.py
# Wrapper script to suppress Gym warnings when testing the environment

import os
import sys

# Set environment variable BEFORE any imports
os.environ["GYM_NOTICES"] = "0"

# Now import and run the environment test
if __name__ == "__main__":
    # Import the environment module
    from energy_env_improved import EnergyMarketEnv
    
    # Quick test
    env = EnergyMarketEnv(n_agents=4, max_line_capacity_kw=200.0, per_agent_max_kw=120.0,
                          base_price=0.12, price_slope=0.002, overload_multiplier=25.0, seed=42)
    obs, _ = env.reset()
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    
    # Run a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.4f}, market_price={info.get('market_price', 0):.4f}")
    
    print("\nEnvironment test completed successfully!")

