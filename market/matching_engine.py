# market/matching_engine.py
import numpy as np

class MatchingEngine:
    """
    Very simple matching engine:
    - Input: orders (array of length n), where positive -> sell (kW), negative -> buy (kW)
    - Matches total supply and demand pro-rata among sellers/buyers.
    - If net supply > demand, grid will absorb surplus (grid_import negative).
    - If net demand > supply, grid will supply remaining (grid_import positive).
    - Computes a simple clearing price using a base price plus supply/demand factor.
    """

    def __init__(self, grid_buy_price=0.20, grid_sell_price=0.10):
        self.grid_buy_price = float(grid_buy_price)   # Price to buy FROM grid (Retail)
        self.grid_sell_price = float(grid_sell_price) # Price to sell TO grid (Feed-in)

    def match(self, bids, grid_buy_price: float = None, grid_sell_price: float = None):
        """
        Matches orders using a Uniform Price Double Auction.
        
        Args:
            bids: np.array of shape (n_agents, 2).
                  Column 0: Quantity (kW), positive=sell, negative=buy.
                  Column 1: Price Limit ($/kWh).
            grid_buy_price: Optional dynamic Retail Price (if None, use default)
            grid_sell_price: Optional dynamic Feed-in Price (if None, use default)
                  
        Returns:
            trades: np.array of shape (n_agents,), actual energy traded (kW)
            clearing_price: float, market clearing price ($/kWh)
            grid_flow: float, net flow with grid
            info: dict
        """
        # Resolve prices
        g_buy = float(grid_buy_price) if grid_buy_price is not None else self.grid_buy_price
        g_sell = float(grid_sell_price) if grid_sell_price is not None else self.grid_sell_price
        bids = np.array(bids, dtype=float)
        quantities = bids[:, 0]
        prices = bids[:, 1]
        
        # Separate Buyers and Sellers
        sellers_idx = np.where(quantities > 1e-6)[0]
        buyers_idx = np.where(quantities < -1e-6)[0]
        
        # Create Order Books
        # Sell Orders: Sort by Price ASC (Cheapest first)
        sell_orders = []
        for idx in sellers_idx:
            sell_orders.append({'id': idx, 'qty': quantities[idx], 'price': prices[idx]})
        sell_orders.sort(key=lambda x: x['price'])
        
        # Buy Orders: Sort by Price DESC (Highest willingness to pay first)
        buy_orders = []
        for idx in buyers_idx:
            buy_orders.append({'id': idx, 'qty': abs(quantities[idx]), 'price': prices[idx]})
        buy_orders.sort(key=lambda x: x['price'], reverse=True)
        
        # --- UNIFORM PRICE AUCTION LOGIC ---
        # We want to find the price P* where Supply(P*) == Demand(P*)
        # Or maximize volume where Bid >= Ask.
        
        # Simplified approach:
        # Stack all bids and asks into a single list of "events" to traverse the curves.
        # But simpler: Iterate matches until spread crosses.
        
        matched_volume = 0.0
        clearing_price = (g_buy + g_sell) / 2.0
        
        # Pointers
        s_i = 0
        b_i = 0
        
        trades = np.zeros(len(bids), dtype=float)
        
        # Track remaining quantities
        current_sell_orders = [s.copy() for s in sell_orders]
        current_buy_orders = [b.copy() for b in buy_orders]
        
        # Find the marginal match
        marginal_price = None
        
        while s_i < len(current_sell_orders) and b_i < len(current_buy_orders):
            seller = current_sell_orders[s_i]
            buyer = current_buy_orders[b_i]
            
            if buyer['price'] >= seller['price']:
                # Match is possible
                qty = min(seller['qty'], buyer['qty'])
                
                # We don't execute yet, we just accumulate volume to find clearing price.
                # In uniform auction, the price is determined by the LAST match (marginal).
                # Usually (Buyer_Last + Seller_Last) / 2
                
                marginal_price = (buyer['price'] + seller['price']) / 2.0
                
                # Update residuals for next iteration
                seller['qty'] -= qty
                buyer['qty'] -= qty
                
                if seller['qty'] < 1e-9:
                    s_i += 1
                if buyer['qty'] < 1e-9:
                    b_i += 1
            else:
                # Spread crossed, no more matches
                break
        
        if marginal_price is not None:
            clearing_price = marginal_price
            
        # --- EXECUTE TRADES AT CLEARING PRICE ---
        # Now we re-iterate and execute all trades that are "in the money" at clearing_price.
        # Sellers with Ask <= Clearing Price
        # Buyers with Bid >= Clearing Price
        
        # Reset residuals
        current_sell_orders = [s.copy() for s in sell_orders]
        current_buy_orders = [b.copy() for b in buy_orders]
        
        # Grid interaction logic needs to be integrated.
        # In a pure P2P pool, we clear internal first.
        # Then residuals go to grid.
        
        # Internal Clearing
        # We match Supply and Demand that met the price criteria.
        # But wait, Supply might != Demand at clearing price due to lumpiness.
        # We need to balance.
        # Usually, the side with excess volume is rationed.
        
        # Let's collect total valid supply and demand at clearing_price
        valid_supply_vol = sum([s['qty'] for s in current_sell_orders if s['price'] <= clearing_price + 1e-9])
        valid_demand_vol = sum([b['qty'] for b in current_buy_orders if b['price'] >= clearing_price - 1e-9])
        
        # The traded volume is min(supply, demand)
        tx_volume = min(valid_supply_vol, valid_demand_vol)
        matched_volume = tx_volume
        
        # Allocate to Sellers (Pro-rata or Priority?)
        # Priority (Cheapest first) is standard.
        # If multiple sellers at same price are marginal, pro-rata.
        # For simplicity: Priority.
        
        # Execute Sells
        remaining_to_sell = tx_volume
        for s in current_sell_orders:
            if s['price'] <= clearing_price + 1e-9 and remaining_to_sell > 0:
                amount = min(s['qty'], remaining_to_sell)
                trades[s['id']] += amount
                remaining_to_sell -= amount
                s['qty'] -= amount # Update residual for grid
        
        # Execute Buys
        remaining_to_buy = tx_volume
        for b in current_buy_orders:
            if b['price'] >= clearing_price - 1e-9 and remaining_to_buy > 0:
                amount = min(b['qty'], remaining_to_buy)
                trades[b['id']] -= amount
                remaining_to_buy -= amount
                b['qty'] -= amount # Update residual for grid
                
        # --- GRID INTERACTION (Residuals) ---
        grid_export = 0.0
        grid_import = 0.0
        
        # Unmatched Sellers -> Grid (if profitable)
        for s in current_sell_orders:
            if s['qty'] > 1e-9:
                # Can sell to grid if Ask <= Feed-in
                # OR if we just dump everything to grid?
                # Usually, if you didn't clear in P2P, you sell to grid at Feed-in.
                # But only if your Ask is low enough?
                # Let's assume yes, if Ask <= Feed-in.
                if s['price'] <= g_sell:
                    trades[s['id']] += s['qty']
                    grid_export += s['qty']
        
        # Unmatched Buyers -> Grid (if willing)
        for b in current_buy_orders:
            if b['qty'] > 1e-9:
                # Buy from grid if Bid >= Retail
                if b['price'] >= g_buy:
                    trades[b['id']] -= b['qty']
                    grid_import += b['qty']
                    
        grid_flow = grid_export - grid_import
        
        # Conservation Check
        # Sum of trades should be equal to grid_flow (net).
        # Sum(trades) = Total_Sold - Total_Bought
        # Total_Sold = P2P_Sold + Grid_Export
        # Total_Bought = P2P_Bought + Grid_Import
        # P2P_Sold == P2P_Bought (by definition of tx_volume)
        # So Sum(trades) should == Grid_Export - Grid_Import = grid_flow
        
        net_trade_sum = np.sum(trades)
        if abs(net_trade_sum - grid_flow) > 1e-5:
            # This is a critical error
            print(f"CRITICAL: Energy Conservation Violation. Net Trades: {net_trade_sum}, Grid Flow: {grid_flow}")
            # Force balance? No, better to warn/fail in tests.
        
        info = {
            "total_volume": matched_volume,
            "clearing_price": clearing_price,
            "grid_flow": grid_flow
        }
        
        return trades, clearing_price, grid_flow, info
