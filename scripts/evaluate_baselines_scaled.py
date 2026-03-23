
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_q1.novelty.slim_env import EnergyMarketEnvSLIM
from baselines.rule_based_agent import RuleBasedAgent

def evaluate_mode(mode, n_agents, n_steps=168, n_seeds=5):
    """
    Evaluates either 'baseline' (grid-only) or 'legacy' (rule-based P2P).
    """
    total_p2p = []
    total_profit = []
    
    for seed in range(n_seeds):
        env = EnergyMarketEnvSLIM(
            n_agents=n_agents,
            data_file="processed_hybrid_data.csv",
            enable_safety=True,
            enable_p2p=(mode != "baseline"),
            market_type=("dynamic" if mode == "slim" else "classic"),
            random_start_day=True,
            diversity_mode=True
        )
        
        obs, _ = env.reset(seed=seed)
        
        # RuleBasedAgent expects a specific observation format.
        # We will wrap it to ensure it gets what it needs or just use its logic.
        agents = [RuleBasedAgent(i, 50.0, 5.0) for i in range(n_agents)] # Approx capacities
        
        seed_p2p = 0
        seed_profit = 0
        
        for t in range(n_steps):
            # 1. Get Actions for each agent
            demands, generations, weather = env._get_current_data()
            retail, feed_in = env._get_grid_prices()
            
            flat_actions = []
            for i in range(n_agents):
                # Construct mock observation for RuleBasedAgent
                # Base dim 8: [Dem, SoC, PV, Exp, Imp, CO2, Retail, FeedIn]
                node = env.nodes[i]
                mock_obs = np.array([demands[i], node.soc, generations[i], 0, 0, 0, retail, feed_in])
                
                # Get rule-based response
                rb_act = agents[i].get_action(mock_obs, t % 24)
                
                # SLIM env expect (N, 2): [BattNormalized, P2PNormalized]
                # Map rb_act [BattkW, P2PkW, Price] to SLIM
                # BattNormalized = BattkW / MaxChargeKw
                # P2PNormalized = P2PkW / 5.0 (SLIM's assumed max P2P rate)
                
                batt_norm = np.clip(rb_act[0] / node.battery_max_charge_kw, -1, 1)
                p2p_norm = np.clip(rb_act[1] / 5.0, -1, 1)
                
                flat_actions.extend([batt_norm, p2p_norm])
                
            # 2. Step Env
            obs, reward, done, truncated, info = env.step(np.array(flat_actions))
            
            # 3. Accumulate Step Metrics
            seed_p2p += info.get("p2p_volume", 0.0)
            # Profit is accumulated in env.accumulated_profit (SLIM specific)
            seed_profit = env.accumulated_profit
            
            if done or truncated:
                break
        
        total_p2p.append(seed_p2p)
        total_profit.append(seed_profit)
        env.close()
        
    return {
        "p2p_mean": np.mean(total_p2p),
        "profit_mean": np.mean(total_profit)
    }

def main():
    scales = [4, 6, 8, 10]
    modes = ["baseline", "legacy"]
    
    results = []
    
    print(f"{'Mode':<15} | {'N':<3} | {'Profit/Agent':<15} | {'P2P/Agent':<15}")
    print("-" * 55)
    
    for mode in modes:
        for n in scales:
            res = evaluate_mode(mode, n, n_steps=500, n_seeds=3) # Reduced for speed
            p_agent = res['profit_mean'] / n
            p2p_agent = res['p2p_mean'] / n
            
            results.append({
                "Mode": mode,
                "N": n,
                "Profit/Agent": p_agent,
                "P2P/Agent": p2p_agent
            })
            
            print(f"{mode:<15} | {n:<3} | {p_agent:>15.2f} | {p2p_agent:>15.2f}")
            
    df = pd.DataFrame(results)
    df.to_csv("evaluation/scalability_baselines.csv", index=False)
    print("\nResults saved to evaluation/scalability_baselines.csv")

if __name__ == "__main__":
    main()
