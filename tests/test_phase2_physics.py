
import unittest
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust

class TestPhase2Physics(unittest.TestCase):
    
    def setUp(self):
        self.ramp_limit = 5.0 # 5 kW/h
        self.resistance = 0.1
        self.voltage = 0.4
        
        self.env = EnergyMarketEnvRobust(
            n_agents=2,
            enable_ramp_rates=True,
            ramp_limit_kw_per_hour=self.ramp_limit,
            enable_losses=True,
            line_resistance_ohms=self.resistance,
            grid_voltage_kv=self.voltage,
            battery_capacity_kwh=50,
            battery_max_charge_kw=20 # Max charge > Ramp limit
        )
        
    def test_ramp_rate_constraint(self):
        """Verify that battery power changes are limited by ramp rate."""
        obs, _ = self.env.reset(seed=42)
        
        # Step 1: Idle (0 kW)
        action_idle = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.env.step(action_idle)
        
        # Check node state - should be 0 (or close)
        node0 = self.env.nodes[0]
        self.assertAlmostEqual(node0.last_power_kw, 0.0)
        
        # Step 2: Request Max Charge (20 kW) -> Should be clipped to 5 kW (Ramp Limit)
        # Action: [Bat, Trade, Price]
        action_max = np.array([20, 0, 0, 20, 0, 0], dtype=np.float32)
        _, _, _, _, info = self.env.step(action_max)
        
        throughput_delta = info["battery_throughput_delta_kwh"][0]
        # Throughput for 1 hour at 5kW = 5kWh (ignoring efficiency for moment or accounting for it)
        # Effective Charge = min(20, Ramp=5) = 5.
        
        print(f"Requested: 20kW. Last: 0kW. Ramp: 5kW. Effective Throughput: {throughput_delta} kWh")
        
        # Depending on efficiency, throughput might be slightly different, but power should be 5.
        # MicrogridNode logic: battery_action_kw is clipped.
        # 5 kW * 1h * sqrt(eff) -> energy.
        
        # Let's inspect the node's last_power_kw
        print(f"Node 0 Last Power: {node0.last_power_kw}")
        self.assertAlmostEqual(node0.last_power_kw, 5.0, delta=0.1)
        
        # Step 3: Request -20 kW (Discharge). Last was +5. Limit is 5.
        # Can go down to 0. (Delta -5).
        # Bounds: [5-5, 5+5] = [0, 10].
        # Wait, if I request -20, and bounds are [0, 10], it should clip to 0?
        # Ramp constraint: current must be within [last - ramp, last + ramp].
        # [0, 10]. Request -20 -> Clip to 0.
        
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
