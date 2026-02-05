import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market.matching_engine import MatchingEngine

class TestMatchingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)

    def test_surplus_scenario(self):
        """Total Sell > Total Buy -> Price should be Grid Sell Price (0.10)"""
        # Agent 0 sells 10 at 0.05, Agent 1 buys 5 at 0.25
        # 10 > 5, so surplus.
        bids = np.array([
            [10.0, 0.05],
            [-5.0, 0.25]
        ])
        
        trades, price, grid_flow, info = self.engine.match(bids)
        
        self.assertAlmostEqual(price, 0.10)
        self.assertAlmostEqual(grid_flow, 5.0) # 5kW export
        self.assertAlmostEqual(trades[0], 10.0)
        self.assertAlmostEqual(trades[1], -5.0)

    def test_deficit_scenario(self):
        """Total Buy > Total Sell -> Price should be Grid Buy Price (0.20)"""
        # Agent 0 sells 5 at 0.05, Agent 1 buys 10 at 0.25
        bids = np.array([
            [5.0, 0.05],
            [-10.0, 0.25]
        ])
        
        trades, price, grid_flow, info = self.engine.match(bids)
        
        self.assertAlmostEqual(price, 0.20)
        self.assertAlmostEqual(grid_flow, -5.0) # 5kW import
        self.assertAlmostEqual(trades[0], 5.0)
        self.assertAlmostEqual(trades[1], -10.0)

    def test_price_mismatch(self):
        """Buyer Bid < Seller Ask -> No Trade"""
        # Agent 0 sells 10 at 0.15
        # Agent 1 buys 10 at 0.12
        # No match possible internally.
        # Seller (0.15) > Grid Sell (0.10) -> No sell to grid.
        # Buyer (0.12) < Grid Buy (0.20) -> No buy from grid.
        # Result: 0 trades.
        
        bids = np.array([
            [10.0, 0.15],
            [-10.0, 0.12]
        ])
        
        trades, price, grid_flow, info = self.engine.match(bids)
        
        self.assertAlmostEqual(np.sum(np.abs(trades)), 0.0)
        self.assertAlmostEqual(grid_flow, 0.0)

if __name__ == '__main__':
    unittest.main()
