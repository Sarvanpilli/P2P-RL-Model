import unittest
import numpy as np
import sys
import os

# Add parent directory to path to find 'train' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.safety_filter import FeasibilityFilter

class TestSafetyFilter(unittest.TestCase):
    def setUp(self):
        self.filter = FeasibilityFilter(
            battery_capacity_kwh=50.0,
            battery_max_charge_kw=25.0,
            timestep_hours=1.0
        )
        self.n_agents = 2
        # State: [demand, soc, pv]
        self.state = np.zeros((self.n_agents, 3))

    def test_soc_max_limit(self):
        """Test that we cannot charge a full battery."""
        # Agent 0 is FULL (50kWh)
        self.state[0, 1] = 50.0
        # Agent 1 is EMPTY (0kWh)
        self.state[1, 1] = 0.0
        
        # Action: Both try to CHARGE 10kW
        raw_action = np.array([[10.0, 0.0], [10.0, 0.0]])
        
        safe_action, corrections = self.filter.filter_action(raw_action, self.state)
        
        # Agent 0 should be clamped to 0
        self.assertEqual(safe_action[0, 0], 0.0)
        # Agent 1 should be allowed to charge
        self.assertEqual(safe_action[1, 0], 10.0)

    def test_soc_min_limit(self):
        """Test that we cannot discharge an empty battery."""
        # Agent 0 is EMPTY
        self.state[0, 1] = 0.0
        # Agent 1 is FULL
        self.state[1, 1] = 50.0
        
        # Action: Both try to DISCHARGE -10kW
        raw_action = np.array([[-10.0, 0.0], [-10.0, 0.0]])
        
        safe_action, corrections = self.filter.filter_action(raw_action, self.state)
        
        # Agent 0 should be clamped to 0
        self.assertEqual(safe_action[0, 0], 0.0)
        # Agent 1 should be allowed to discharge
        self.assertEqual(safe_action[1, 0], -10.0)

    def test_partial_charge(self):
        """Test that we can only charge up to capacity."""
        # Agent 0 has 45kWh (Capacity 50). Can accept 5kWh.
        self.state[0, 1] = 45.0
        
        # Action: Try to charge 10kW (10kWh in 1 hour)
        raw_action = np.array([[10.0, 0.0], [0.0, 0.0]])
        
        safe_action, corrections = self.filter.filter_action(raw_action, self.state)
        
        # Should be clamped to 5.0
        self.assertAlmostEqual(safe_action[0, 0], 5.0)

if __name__ == '__main__':
    unittest.main()
