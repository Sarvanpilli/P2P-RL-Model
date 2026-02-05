
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

from train.energy_env_robust import EnergyMarketEnvRobust

def verify_profit():
    print("--- Verifying Positive Reward Scenario ---")
    
    # 1. Initialize
    env = EnergyMarketEnvRobust(n_agents=4)
    env.reset()
    
    # 2. Force State: "Solar Surplus"
    # Agent 0: Demand=0, SoC=20, PV=50
    # Ideally should sell ~50kW.
    env.state[0] = [0.0, 20.0, 50.0]
    
    print(f"State set to: Demand={env.state[0][0]}, SoC={env.state[0][1]}, PV={env.state[0][2]}")
    
    # 3. Action: Sell 40kW @ $0.15/kWh
    # Action: [Battery_kW, Trade_kW, Price]
    # Trade > 0 is Sell (Export) in our convention?
    # Let's check environment convention. Usually +Trade = Export.
    
    actions = np.zeros((4, 3), dtype=np.float32)
    actions[0] = [0.0, 40.0, 0.15] 
    
    print(f"Action: Battery={actions[0][0]}, Trade={actions[0][1]} (Sell), Price={actions[0][2]}")
    
    # 4. Step
    obs, reward, terminated, truncated, info = env.step(actions.flatten())
    
    print("\n--- Result ---")
    print(f"Total Reward: {reward:.4f}")
    
    # Check granular info if available
    # RewardTracker logs are usually in info.
    # Looking for 'reward/profit_mean' or similar if aggregated, or we can inspect tracker directly.
    tracker = env.reward_tracker
    agent_profit = tracker.step_profit[0]
    print(f"Agent 0 Financial Profit: ${agent_profit:.4f}")
    
    if reward > 0 and agent_profit > 0:
        print("\nSUCCESS: Positive Reward Achieved!")
    else:
        print("\nFAILURE: Reward is not positive. Check penalties.")
        print(f"Penalties: CO2={tracker.step_co2_penalty[0]}, SoC={tracker.step_soc_penalty[0]}, Grid={tracker.step_grid_penalty[0]}")

if __name__ == "__main__":
    verify_profit()
