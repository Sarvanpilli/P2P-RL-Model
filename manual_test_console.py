
import numpy as np
import sys
import os
import argparse
from stable_baselines3 import PPO

# Add parent dir to path
sys.path.append(os.getcwd())

try:
    from train.energy_env_robust import EnergyMarketEnvRobust
except ImportError:
    print("Error: Could not import EnergyMarketEnvRobust. Run this from proejct root.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Interactive Manual Test Console")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (optional)")
    args = parser.parse_args()

    print("--- P2P Energy Trading: Interactive Console ---")
    print("Initializing Environment...")
    env = EnergyMarketEnvRobust(n_agents=4)
    obs, info = env.reset()
    
    model = None
    if args.model and os.path.exists(args.model):
        print(f"Loading model from {args.model}...")
        model = PPO.load(args.model)
    elif args.model:
        print(f"Warning: Model file {args.model} not found.")

    while True:
        print("\n" + "="*50)
        print(f"Timestep: {env.timestep_count} | Hour: {env.timestep_count % 24}:00")
        print("-" * 50)
        
        # Display Current State (Agent 0 for simplicity, or Summary)
        # State: [Demand, SoC, PV]
        state = env.state
        print("Current State (Agent 0 only):")
        print(f"  Demand: {state[0][0]:.2f} kW")
        print(f"  SoC:    {state[0][1]:.2f} kWh (Max {env.battery_capacity_kwh})")
        print(f"  PV:     {state[0][2]:.2f} kW")
        
        # Input Loop
        cmd = input("\nCommands: [n]ext step (model), [m]anual action, [s]et state, [q]uit: ").strip().lower()
        
        if cmd == 'q':
            break
            
        elif cmd == 's':
            # Set State Manually
            try:
                print("\nSet State for Agent 0:")
                d = float(input("  Demand (kW): ") or state[0][0])
                s = float(input("  SoC (kWh): ") or state[0][1])
                p = float(input("  PV (kW): ") or state[0][2])
                
                env.state[0] = [d, s, p]
                print("State updated.")
            except ValueError:
                print("Invalid input.")
                continue

        elif cmd == 'm':
            # Manual Action
            try:
                print("\nInput Action for Agent 0 (Others will be idle):")
                batt = float(input("  Battery (+Ch/-Dis) kW: ") or 0.0)
                trade = float(input("  Trade (+Sell/-Buy) kW: ") or 0.0)
                price = float(input("  Price Bid ($/kWh): ") or 0.10)
                
                # Construct action array
                # Actions: [Batt, Trade, Price]
                actions = np.zeros((4, 3), dtype=np.float32)
                actions[0] = [batt, trade, price]
                
                # Step
                obs, reward, terminated, truncated, info = env.step(actions.flatten())
                print_step_info(info, reward)
                
            except ValueError:
                print("Invalid input.")
                continue

        elif cmd == 'n':
            # Model Action
            if model:
                action, _ = model.predict(obs, deterministic=True)
                print("\nModel Prediction Executed.")
            else:
                print("\nNo model loaded. Using Random Action.")
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            print_step_info(info, reward)
            
def print_step_info(info, reward):
    print("\n--- Step Results ---")
    print(f"Total Reward: {reward:.4f}")
    
    if "safety_violations" in info:
        print(f"Safety Violations: {info['safety_violations']}")
    if "fallback_triggered" in info:
        print(f"Fallback Triggered: {info['fallback_triggered']}")
        
    # Check specifics for Agent 0 if available in info
    # Usually info is aggregated or per-step.
    # We can check specific keys added by Guard
    print(f"Info keys: {list(info.keys())}")

if __name__ == "__main__":
    main()
