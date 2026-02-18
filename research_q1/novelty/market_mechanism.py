
import numpy as np

class LiquidityPool:
    """
    Layer 3: Liquidity Matching Mechanism
    Aggregates P2P bids and clears them at a fair mid-market price.
    Uses Pro-Rata allocation for imbalances.
    """
    def __init__(self, 
                 grid_buy_price: float = 0.15,
                 grid_sell_price: float = 0.05):
        self.grid_buy = grid_buy_price
        self.grid_sell = grid_sell_price
        
    def update_prices(self, grid_buy: float, grid_sell: float):
        """Update current grid reference prices."""
        self.grid_buy = grid_buy
        self.grid_sell = grid_sell

    def clear_market(self, orders: list):
        """
        Clears the market based on aggregated Supply and Demand.
        
        Args:
            orders: List of floats. 
                    Positive = Buy Request (kWh)
                    Negative = Sell Offer (kWh)
                    
        Returns:
            cleared_quantities: List of floats (signed)
            clearing_price: float
            stats: dict
        """
        orders = np.array(orders)
        
        # 1. Separate Supply and Demand
        # Demand: Sum of positive orders
        total_demand = np.sum(orders[orders > 0])
        # Supply: Sum of absolute negative orders
        total_supply = np.abs(np.sum(orders[orders < 0]))
        
        # 2. Determine Clearing Price
        # Strategy: Split the spread (Mid-Market)
        # This incentivizes P2P over Grid for both sides.
        clearing_price = (self.grid_buy + self.grid_sell) / 2.0
        
        # 3. Match Liquidity
        # Volume traded is limited by the smaller side
        traded_volume = min(total_demand, total_supply)
        
        # 4. Calculate Match Ratios (Pro-Rata)
        if total_demand > 0:
            buy_ratio = traded_volume / total_demand
        else:
            buy_ratio = 0.0
            
        if total_supply > 0:
            sell_ratio = traded_volume / total_supply
        else:
            sell_ratio = 0.0
            
        # 5. Execute Trades
        cleared_orders = np.zeros_like(orders)
        
        for i, order in enumerate(orders):
            if order > 0: # Buyer
                cleared_orders[i] = order * buy_ratio
            elif order < 0: # Seller
                # cleared is negative to indicate export
                cleared_orders[i] = order * sell_ratio
            else:
                cleared_orders[i] = 0.0
                
        stats = {
            'total_demand': total_demand,
            'total_supply': total_supply,
            'traded_volume': traded_volume,
            'clearing_price': clearing_price,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio
        }
        
        return cleared_orders, clearing_price, stats
