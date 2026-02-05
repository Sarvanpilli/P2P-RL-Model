from market.matching_engine import MatchingEngine
import numpy as np

def test_volatility():
    print("--- Market Volatility Stress Test ---")
    me = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)
    
    # Scene 1: Extreme Surplus (Sunny Day)
    # Everyone selling. Demands are low.
    # Expectation: Price crashes to Feed-in (0.10)
    bids_surplus = np.array([
        [10.0, 0.09], # Sell 10kW, willing to take 0.09
        [10.0, 0.10],
        [10.0, 0.11],
        [10.0, 0.12]
    ])
    _, price_surplus, _, _ = me.match(bids_surplus)
    print(f"Scenario 1 (Surplus): Clearing Price = ${price_surplus:.2f} (Expected ~0.10)")

    # Scene 2: Extreme Shortage (Night)
    # Everyone buying. No PV.
    # Expectation: Price spikes to Retail (0.20) or higher (if they bid high)
    bids_shortage = np.array([
        [-10.0, 0.25], # Buy 10kW, willing to pay 0.25
        [-10.0, 0.22],
        [-10.0, 0.21],
        [-10.0, 0.19]
    ])
    _, price_shortage, _, _ = me.match(bids_shortage)
    print(f"Scenario 2 (Shortage): Clearing Price = ${price_shortage:.2f} (Expected ~0.20+)")
    
    # Scene 3: Balanced Trading
    # 2 Buyers, 2 Sellers overlapping
    bids_balanced = np.array([
        [5.0, 0.12],  # Sell @ 0.12
        [5.0, 0.14],  # Sell @ 0.14
        [-5.0, 0.18], # Buy  @ 0.18
        [-5.0, 0.16]  # Buy  @ 0.16
    ])
    # Match: Seller 0.12 vs Buyer 0.18 -> Avg 0.15
    # Match: Seller 0.14 vs Buyer 0.16 -> Avg 0.15
    _, price_balanced, _, _ = me.match(bids_balanced)
    print(f"Scenario 3 (Balanced): Clearing Price = ${price_balanced:.2f} (Expected ~0.15)")

    if price_surplus < price_balanced < price_shortage:
        print("\nSUCCESS: Market responds dynamically to Supply/Demand!")
    else:
        print("\nFAILURE: Prices are static/broken.")

if __name__ == "__main__":
    test_volatility()
