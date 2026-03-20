
import unittest
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust

class TestPhase2Physics(unittest.TestCase):
    
    def setUp(self):
        self.ramp_limit = 1.0 # 1.0 kW/h, must be < 2.5 (default battery bounds)
        self.resistance = 0.1
        self.voltage = 0.4
        
        self.env = EnergyMarketEnvRobust(
            n_agents=2,
            enable_ramp_rates=True,
            enable_losses=True
        )
        self.env.line_resistance_ohms = self.resistance
        self.env.grid_voltage_kv = self.voltage
        
        for n in self.env.nodes:
            n.set_ramp_limit(self.ramp_limit)
        
    def test_ramp_rate_constraint(self):
        """Verify that battery power changes are limited by ramp rate."""
        obs, _ = self.env.reset(seed=42)
        
        # Step 1: Idle (0 kW)
        action_idle = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.env.step(action_idle)
        
        # Check node state - should be 0 (or close)
        node0 = self.env.nodes[0]
        self.assertAlmostEqual(node0.last_power_kw, 0.0)
        
        # Step 2: Request Max Charge (20 kW) -> Should be clipped by action space to 2.5, then by ramp limit to 1.0
        # Action: [Bat, Trade, Price]
        action_max = np.array([20, 0, 0, 20, 0, 0], dtype=np.float32)
        _, _, _, _, info = self.env.step(action_max)
        
        # Let's inspect the node's last_power_kw
        print(f"Node 0 Last Power: {node0.last_power_kw}")
        self.assertAlmostEqual(node0.last_power_kw, 1.0, delta=0.1)
        
        # Step 3: Request -20 kW (Discharge). Last was +1. Limit is 1.
        # Can go down to 0. (Delta -1).
        
        action_discharge = np.array([-20, 0, 0, -20, 0, 0], dtype=np.float32)
        self.env.step(action_discharge)
        print(f"Node 0 Last Power (Expect 0): {node0.last_power_kw}")
        self.assertAlmostEqual(node0.last_power_kw, 0.0, delta=0.1)

    def test_distribution_losses(self):
        """Verify that losses are non-zero when power flows."""
        # Reset
        self.env.reset(seed=42)
        
        # Force a scenario where agents have net load.
        # e.g. Agent 0 Charges 5kW. Agent 1 Charges 5kW.
        # Total network flow = 10kW.
        # Loss = (10 / 0.4)^2 * 0.1 / 1000 = (25)^2 * 0.0001 = 625 * 0.0001 = 0.0625 kW.
        
        # We need to bypass the "Feasibility" (Guard) slightly or ensure we have capacity.
        # 50kWh battery, initialized at ~50%. We can charge.
        
        # Action: Charge 5kW (within ramp)
        action = np.array([5.0, 0, 0, 5.0, 0, 0], dtype=np.float32)
        
        # We need to ensure Demand/PV doesn't mess up our specific calc.
        # But we can just check if Loss > 0 and roughly consistent.
        
        _, _, _, _, info = self.env.step(action)
        
        loss = info["loss_kw"]
        print(f"Total Loss reported: {loss} kW")
        
        self.assertGreater(loss, 0.0)
        
        # Rough check
        # Net Load will include Demand - PV.
        # We don't know exact Demand/PV without checking env.
        # But we know loss should correspond to Total Import/Export + Internal?
        # Our formula: Sum(|NetLoad|).
        # We can check consistence.
        
if __name__ == '__main__':
    unittest.main()
