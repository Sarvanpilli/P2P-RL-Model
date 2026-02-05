import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from train.energy_env_robust import EnergyMarketEnvRobust
from market.matching_engine import MatchingEngine

def test_improvements():
    print("Initializing Environment...")
    env = EnergyMarketEnvRobust(n_agents=4, forecast_horizon=1)
    
    # Check Observation Space
    obs_dim = env.observation_space.shape[0] // 4
    print(f"Obs Dim per agent: {obs_dim}")
    # Expected: 8 + 4*1 = 12
    assert obs_dim == 12, f"Expected 12, got {obs_dim}"
    
    obs, _ = env.reset(seed=42)
    print("Reset successful.")
    
    # Check Price Logic
    retail, feed_in = env._get_grid_prices()
    print(f"Prices at step 0: Retail={retail}, FeedIn={feed_in}")
    
    # Take a Step
    action = env.action_space.sample()
    # Force some high bids
    action[2] = 10.0 # Agent 0 price bid
    
    print("Stepping...")
    n_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Market Price: {info['market_price']}")
    print(f"Total Carbon: {info['total_carbon_mass']}")
    print(f"Reward: {reward}")
    
    print("ALL CHECKS PASSED.")

if __name__ == "__main__":
    test_improvements()
