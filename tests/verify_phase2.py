
import numpy as np
import sys
import os

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_advanced import EnergyMarketEnvAdvanced

def test_phase2():
    print("Initializing Advanced Environment...")
    env = EnergyMarketEnvAdvanced(
        n_agents=4, 
        ramp_limit_kw_per_hour=10.0, # Strict ramp
        grid_resistance_ohms=0.1,
        step_hours=1.0
    )
    
    # 1. Check Guard Type
    print(f"Guard Type: {type(env.guard)}")
    assert "ActiveGuard" in str(type(env.guard)), "Guard should be ActiveGuard"
    
    # 2. Check Ramp Constraint
    obs, _ = env.reset(seed=42)
    # Get a node
    node = env.nodes[0]
    node.run_checks = True # theoretical flag
    
    # Force state
    node.last_power_kw = 0.0
    node.set_ramp_limit(5.0) # 5 kW/h
    
    # Try to charge 20 kW
    res = node.step(
        battery_action_kw=20.0,
        current_demand_kw=0,
        current_pv_kw=0,
        dt_hours=1.0
    )
    eff_charge = res['effective_charge']
    print(f"Requested 20kW, Ramp Limit 5kW -> Effective: {eff_charge}")
    assert eff_charge <= 5.0 + 1e-5, f"Ramp failed, got {eff_charge}"
    
    # 3. Check Losses
    # Reset
    env.reset()
    # Mock matching engine to force trade
    # We can't easily mock internal engine without patching.
    # Instead, we force an action that causes high trade.
    
    # Action: Agent 0 Sells 100, Agent 1 Buys 100
    # Batts are limited, so maybe we can't force 100.
    # But Grid interactions?
    # Advanced Env calculates loss on Net Flow.
    
    action = env.action_space.sample() * 0 # No battery
    # But we want to test _step_grid_and_reward.
    # Let's step and check info.
    
    obs, reward, done, trunc, info = env.step(action)
    print(f"Loss KW in info: {info.get('dist_loss_total')}")
    assert 'dist_loss_total' in info
    
    # 4. Check Active Guard Stats
    # Run a few steps
    for _ in range(5):
        env.step(env.action_space.sample())
        
    print(f"Guard Stats Count: {env.guard.count}")
    assert env.guard.count > 0, "Guard should update stats"
    
    print("ALL PHASE 2 CHECKS PASSED.")

if __name__ == "__main__":
    test_phase2()
