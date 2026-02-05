import numpy as np
import matplotlib.pyplot as plt
from market.matching_engine import MatchingEngine

def generate_and_plot():
    print("Generating comparison data...")
    me = MatchingEngine(grid_buy_price=0.20, grid_sell_price=0.10)
    
    # Time steps (0 to 23 hours)
    hours = np.arange(24)
    
    prices_surplus = []
    prices_shortage = []
    
    # 1. Simulate Surplus Day (Sunny)
    # Scenario: High Supply (Sellers) > Low Demand (Buyers)
    # Result: Price crashes to clear the excess.
    for h in hours:
        bids = np.array([
            [2.0, 0.09],  # Seller A: Desperate to sell
            [2.0, 0.09],  # Seller B
            [2.0, 0.10],  # Seller C
            [-1.0, 0.11]  # Buyer D: Opportunistic, but only wants a little
        ])
        # Match: S(0.09) vs B(0.11) -> Price ~0.10
        _, price, _, _ = me.match(bids)
        prices_surplus.append(price)

    # 2. Simulate Shortage Day (Winter Night / Storm)
    # Scenario: High Demand (Buyers) >> Scarce Supply (Sellers)
    # Result: Buyers bid up the price to secure the rare energy.
    for h in hours:
        bids = np.array([
            [-2.0, 0.22], # Buyer A: Desperate (Willing to pay > Grid)
            [-2.0, 0.24], # Buyer B: Very Desperate
            [-2.0, 0.25], # Buyer C: Extremely Desperate
            [1.0, 0.20]   # Seller D: Greedy (Holding out for high price)
        ])
        # Match: B(0.25) vs S(0.20) -> Price = (0.25+0.20)/2 = 0.225
        _, price, _, _ = me.match(bids)
        prices_shortage.append(price)

    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Surplus (Current)
    plt.plot(hours, prices_surplus, label="Scenario A: Sunny Day (Surplus)", color="green", linewidth=3, linestyle="--")
    
    # Plot Shortage (New)
    plt.plot(hours, prices_shortage, label="Scenario B: Winter Storm (Shortage)", color="red", linewidth=3)
    
    # Reference Lines
    plt.axhline(y=0.20, color='gray', linestyle=':', alpha=0.5, label="Grid Retail Price ($0.20)")
    plt.axhline(y=0.10, color='gray', linestyle=':', alpha=0.5, label="Grid Feed-in Price ($0.10)")
    
    plt.title("Market Price Comparison: Why volatility matters", fontsize=14)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Market Price ($/kWh)", fontsize=12)
    plt.ylim(0, 0.30)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    output_path = "evaluation/market_comparison.png"
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    generate_and_plot()
