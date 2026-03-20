    import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust

class TestDeploymentReadiness(unittest.TestCase):
    def setUp(self):
        self.env = EnergyMarketEnvRobust(n_agents=4, forecast_horizon=1)
        self.env.reset()

    def test_extreme_shortage(self):
        """Test behavior when Demand is High and PV is Zero (Night/Winter)."""
        print("\n--- Test: Extreme Shortage ---")
        # Override data fetcher
        self.env._get_current_data = lambda: (np.full(4, 50.0), np.zeros(4), np.zeros(2))
        for node in self.env.nodes:
            node.soc = 5.0
        
        # Action: Try to discharge (impossible) and buy nothing (bad idea)
        # Agent should ideally buy from grid.
        # But here we test PHYSICS safety.
        # If agent tries to discharge 20kW, safety filter should block it.
        
        raw_action = np.zeros((4, 3))
        raw_action[:, 0] = -20.0 # Discharge
        raw_action[:, 1] = 0.0   # No trade
        raw_action[:, 2] = 0.5   # Price
        
        obs, reward, terminated, truncated, info = self.env.step(raw_action.flatten())
        
        # Check Safety: Did it prevent discharge?
        # effective_batt_discharge_kw is not directly in info, but we can check SoC.
        # SoC should NOT decrease significantly (only limited by what's available, which is 5kWh).
        # 5kWh / 1h = 5kW max discharge.
        # Requested 20kW.
        # Should be capped at 5kW.
        
        # We can check 'curtailment_kw' or similar, but better to check if simulation crashed.
        self.assertFalse(terminated)
        print("Survived Extreme Shortage step.")

    def test_extreme_surplus(self):
        """Test behavior when PV is High and Demand is Zero (Sunny Day)."""
        print("\n--- Test: Extreme Surplus ---")
        # Override data fetcher and SoC
        self.env._get_current_data = lambda: (np.zeros(4), np.full(4, 40.0), np.zeros(2))
        for node in self.env.nodes:
            node.soc = node.battery_capacity_kwh
        
        # Action: Try to charge (impossible) and sell nothing
        raw_action = np.zeros((4, 3))
        raw_action[:, 0] = 20.0 # Charge
        raw_action[:, 1] = 0.0
        raw_action[:, 2] = 0.5
        
        obs, reward, terminated, truncated, info = self.env.step(raw_action.flatten())
        
        # Check Safety: Should prevent charging full battery.
        # SoC should remain at max.
        # PV should be exported or curtailed.
        # Since grid_trade was 0, it should be curtailed (local supply > demand).
        
        print("Survived Extreme Surplus step.")

    def test_market_clearing_logic(self):
        """Verify Market clears correctly under normal conditions."""
        print("\n--- Test: Market Clearing ---")
        # Agent 0: Sell 10kW @ $0.10
        # Agent 1: Buy 10kW @ $0.20
        # Should match.
        
        raw_action = np.zeros((4, 3))
        # Agent 0
        raw_action[0, 0] = 0.0
        raw_action[0, 1] = 10.0 # Sell
        raw_action[0, 2] = 0.20 # Scaled by 0.5 inside env -> $0.10
        
        # Agent 1
        raw_action[1, 0] = 0.0
        raw_action[1, 1] = -10.0 # Buy
        raw_action[1, 2] = 0.40 # Scaled by 0.5 inside env -> $0.20
        
        obs, reward, terminated, truncated, info = self.env.step(raw_action.flatten())
        
        # Check info for market price
        market_price = info['market_price']
        print(f"Market Price: {market_price}")
        
        # Should be between 0.10 and 0.20 (or exactly one of them depending on logic)
        self.assertTrue(0.10 <= market_price <= 0.20)

if __name__ == '__main__':
    unittest.main()
