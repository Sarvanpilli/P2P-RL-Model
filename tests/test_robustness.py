import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
# from train.energy_env_improved import EnergyMarketEnv # Legacy
from train.safety_filter import FeasibilityFilter
from market.matching_engine import MatchingEngine

class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.env = EnergyMarketEnvRobust(n_agents=4)
        self.filter = FeasibilityFilter()
        self.engine = MatchingEngine()

    def test_filter_idempotence(self):
        """Test that applying the filter twice yields the same result."""
        print("\n--- Test: Filter Idempotence ---")
        state = np.array([
            [10.0, 25.0, 5.0], # Demand, SoC, PV
            [5.0, 45.0, 0.0],
            [0.0, 5.0, 20.0],
            [20.0, 10.0, 0.0]
        ])
        # Random raw action
        raw_action = np.array([
            [100.0, 100.0, 1.5], # Invalid: High Charge, High Export, High Price
            [-100.0, -100.0, -0.5], # Invalid: High Discharge, High Import, Low Price
            [10.0, 0.0, 0.5],
            [0.0, 0.0, 0.5]
        ])
        
        safe_action_1, changed_1 = self.filter.filter_action(raw_action, state)
        safe_action_2, changed_2 = self.filter.filter_action(safe_action_1, state)
        
        # Check equality
        np.testing.assert_array_almost_equal(safe_action_1, safe_action_2)
        # Second pass should not change anything
        self.assertFalse(changed_2)
        print("Filter is Idempotent.")

    def test_energy_conservation_matching(self):
        """Test that the matching engine conserves energy."""
        print("\n--- Test: Matching Conservation ---")
        # Scenario:
        # Agent 0: Sell 10 @ 0.12
        # Agent 1: Buy 5 @ 0.15
        # Agent 2: Buy 8 @ 0.14
        # Agent 3: Sell 5 @ 0.11
        
        bids = np.array([
            [10.0, 0.12],
            [-5.0, 0.15],
            [-8.0, 0.14],
            [5.0, 0.11]
        ])
        
        trades, price, grid_flow, info = self.engine.match(bids)
        
        net_trades = np.sum(trades)
        print(f"Net Trades: {net_trades}, Grid Flow: {grid_flow}")
        
        # Conservation: Net Trades should equal Grid Flow
        self.assertAlmostEqual(net_trades, grid_flow, places=5)
        print("Energy Conserved in Matching.")

    def test_physics_balance(self):
        """Test that environment physics conserves energy."""
        print("\n--- Test: Env Physics Balance ---")
        self.env.reset()
        # Force a state
        self.env.state[:, 1] = 25.0 # 50% SoC
        
        # Action: Charge and Discharge
        action = np.zeros((4, 3))
        action[0, 0] = 10.0 # Charge
        action[1, 0] = -10.0 # Discharge
        
        try:
            obs, reward, terminated, truncated, info = self.env.step(action.flatten())
        except Exception as e:
            print(f"Step failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Step failed: {e}")
        
        # Check SoC changes
        # Agent 0: Charged 10kW for 1h = 10kWh (ignoring eff for rough check)
        # Agent 1: Discharged 10kW for 1h = 10kWh
        
        # Exact check with efficiency
        # In Env: eff_charge = self.battery_eff ** 0.5
        # In Env: energy_charged = power * dt * eff_charge
        # In Env: energy_discharged = power * dt / eff_discharge
        
        # Check what self.battery_eff is in env
        eff = self.env.battery_eff
        eff_sqrt = eff ** 0.5
        
        expected_soc_0 = 25.0 + 10.0 * 1.0 * eff_sqrt
        expected_soc_1 = 25.0 - 10.0 * 1.0 / eff_sqrt
        
        self.assertAlmostEqual(self.env.state[0, 1], expected_soc_0, places=4)
        self.assertAlmostEqual(self.env.state[1, 1], expected_soc_1, places=4)
        print("Physics Balance Verified.")

if __name__ == '__main__':
    unittest.main()
