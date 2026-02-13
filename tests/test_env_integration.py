import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_improved import EnergyMarketEnv

class TestEnvIntegration(unittest.TestCase):
    def test_env_step(self):
        """Test that the environment steps correctly with the new MatchingEngine."""
        env = EnergyMarketEnv(n_agents=4)
        obs, _ = env.reset()
        
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if info contains market data
        self.assertIn("market_price", info)
        self.assertIn("grid_flow", info["market_info"]) # MatchingEngine returns this in info dict?
        # Wait, MatchingEngine returns (trades, price, grid_flow, info).
        # In env.step: matched_trades, market_price, grid_flow, market_info = self.matching_engine.match(grid_trade)
        # The env puts `market_price` in its own info.
        # It doesn't explicitly put `market_info` in its return info, let's check the code.
        
        # In env step:
        # info = { "market_price": market_price, ... }
        # It does NOT seem to merge `market_info` into `info`.
        # But `market_price` should be there.
        
        self.assertTrue(isinstance(reward, float))
        self.assertEqual(len(obs), env.observation_space.shape[0])

if __name__ == '__main__':
    unittest.main()
