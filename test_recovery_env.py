"""Quick test script to verify recovery environment works"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing EnergyMarketEnvRecovery...")

try:
    from train.energy_env_recovery import EnergyMarketEnvRecovery
    print("✓ Import successful")
    
    print("\nCreating environment...")
    env = EnergyMarketEnvRecovery(
        n_agents=4,
        data_file="processed_hybrid_data.csv",
        random_start_day=False,
        forecast_horizon=4
    )
    print("✓ Environment created")
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Expected: ({4 * 22},) = (88,)")
    
    print("\nTesting step...")
    action = env.action_space.sample()
    print(f"  Action shape: {action.shape}")
    print(f"  Expected: ({4 * 2},) = (8,)")
    
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  P2P Volume: {info.get('p2p_volume', 0):.4f} kW")
    print(f"  Mean SoC: {info.get('mean_soc_pct', 0):.2%}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
