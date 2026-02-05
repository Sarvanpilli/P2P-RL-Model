
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.energy_env_advanced import EnergyMarketEnvAdvanced

def test_short_run():
    print("Initializing...")
    env = EnergyMarketEnvAdvanced(n_agents=4, ramp_limit_kw_per_hour=50.0)
    env.reset()
    
    print("Starting loop...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if i % 10 == 0:
            print(f"Step {i} done. Reward: {reward:.2f}")
            
    print("Done!")

if __name__ == "__main__":
    test_short_run()
