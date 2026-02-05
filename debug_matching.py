from market.matching_engine import MatchingEngine
import numpy as np

def test_engine():
    me = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)
    
    # Scene 1: No trades (Everyone idle)
    bids = np.zeros((4, 2))
    trades, price, flow, _ = me.match(bids)
    print(f"Scene 1 (Idle): Price={price} (Expected 0.15), Flow={flow}")
    
    # Scene 2: All Buyers (Deficit)
    # 4 agents, all want to buy 1kW at price 0.22
    bids = np.array([
        [-1.0, 0.22],
        [-1.0, 0.22],
        [-1.0, 0.22],
        [-1.0, 0.22]
    ])
    trades, price, flow, _ = me.match(bids)
    print(f"Scene 2 (All Buy): Price={price} (Expected 0.15?), Flow={flow}")
    
    # Scene 3: All Sellers (Surplus)
    # 4 agents, all want to sell 1kW at price 0.08
    bids = np.array([
        [1.0, 0.08],
        [1.0, 0.08],
        [1.0, 0.08],
        [1.0, 0.08]
    ])
    trades, price, flow, _ = me.match(bids)
    print(f"Scene 3 (All Sell): Price={price} (Expected 0.15?), Flow={flow}")

    # Scene 4: Actual CSV Case
    # Agent 0: Buy? (Grid Trade 0)
    # One Agent Sell, One Agent Buy?
    # Let's try mixed
    bids = np.array([
        [-1.0, 0.22], # Buy
        [1.0, 0.08]   # Sell cheap
    ])
    trades, price, flow, _ = me.match(bids)
    print(f"Scene 4 (Mix): Price={price} (Expected ~0.15), Flow={flow}")

if __name__ == "__main__":
    test_engine()
