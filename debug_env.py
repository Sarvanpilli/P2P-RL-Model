import numpy as np
import os
import sys

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from train.energy_env_robust import EnergyMarketEnvRobust

def debug():
    print("Debugging EnergyMarketEnvRobust (Zero Action Accountancy)...")
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file="evaluation/ausgrid_p2p_energy_dataset.csv",
        random_start_day=False
    )
    
    obs, info = env.reset()
    demands, pvs, co2 = env._get_current_data()
    print(f"Step 0 Data: Demands={demands}, PVs={pvs}")
    
    # Action: ALL ZERO
    action = np.zeros((4, 3))
    action_flat = action.flatten()
    
    obs, reward, done, trunc, info = env.step(action_flat)
    
    print("\nStep 1 Result (Zero Actions):")
    print(f"  P2P Volume (Step): {info.get('p2p_volume_kwh_step', 0.0)}")
    print(f"  Total Export: {info.get('total_export', 0.0)}")
    print(f"  Total Import: {info.get('total_import', 0.0)}")
    
    # Expected: Total Import should be roughly sum(demands)
    expected_import = np.sum(demands) - np.sum(pvs)
    print(f"  Expected Net Import: ~{max(0, expected_import):.4f}")
    
if __name__ == "__main__":
    debug()
