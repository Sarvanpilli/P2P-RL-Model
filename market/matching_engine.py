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

    def match(self, bids, grid_buy_price: float = None, grid_sell_price: float = None, 
              epsilon: float = 0.12, margin: float = 0.0, progress: float = 1.0,
              use_curriculum: bool = True):
        """
        Matches orders using a Uniform Price Double Auction.
        """
        # --- PHASE 13: SCIENTIFIC VALIDATION ABLATION ---
        if not use_curriculum:
            effective_epsilon = 0.10
            effective_margin = 0.0
        else:
            # --- PHASE 12: MATCH EXPANSION ---
            # During early training (progress < 0.3), we allow extremely loose matches
            effective_epsilon = epsilon
            effective_margin = margin
            if progress < 0.3:
                effective_epsilon += 0.1
                effective_margin += 0.05
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
        sell_orders = []
        for idx in sellers_idx:
            sell_orders.append({'id': idx, 'qty': quantities[idx], 'price': prices[idx]})
        sell_orders.sort(key=lambda x: x['price'])
        
        buy_orders = []
        for idx in buyers_idx:
            buy_orders.append({'id': idx, 'qty': abs(quantities[idx]), 'price': prices[idx]})
        buy_orders.sort(key=lambda x: x['price'], reverse=True)
        
        # --- FLEXIBLE MID-PRICE AUCTION LOGIC ---
        current_sell_orders = [s.copy() for s in sell_orders]
        current_buy_orders = [b.copy() for b in buy_orders]
        
        # Match Pairs based on Spread (Greedy Volume Maximization)
        # PRIORITIZING TIGHTEST SPREADS FIRST
        # Epsilon and Margin are now dynamic parameters from Step 1/2 of curriculum.
        
        possible_pairs = []
        for b in current_buy_orders:
            for s in current_sell_orders:
                spread = b['price'] - s['price']
                # Soft matching margin applied (Phase 12 effective relaxation)
                if spread >= -(effective_epsilon + effective_margin):
                    possible_pairs.append({
                        'b_idx': b['id'],
                        's_idx': s['id'],
                        'spread': spread,
                        'b_ord': b,
                        's_ord': s
                    })
        
        # Sort by absolute spread (ascending) to prioritize tightest matchups first
        possible_pairs.sort(key=lambda x: abs(x['spread']))
        
        trades = np.zeros(len(bids), dtype=float)
        matched_volume = 0.0
        all_matches = []
        
        for pair in possible_pairs:
            buyer = pair['b_ord']
            seller = pair['s_ord']
            
            if buyer['qty'] > 1e-9 and seller['qty'] > 1e-9:
                qty = min(buyer['qty'], seller['qty'])
                potential_price = (buyer['price'] + seller['price']) / 2.0
                
                # PHASE 12: STRICT PRICE VALIDATION (RELAXED DURING CURRICULUM)
                # seller_min_price - margin <= clearing_price <= buyer_max_price + margin
                if potential_price <= buyer['price'] + effective_margin + 1e-9 and potential_price >= seller['price'] - effective_margin - 1e-9:
                    trades[seller['id']] += qty
                    trades[buyer['id']] -= qty
                    matched_volume += qty
                    all_matches.append((buyer['id'], seller['id'], qty, potential_price))
                    
                    buyer['qty'] -= qty
                    seller['qty'] -= qty
        
        # Determine average clearing price for info
        if matched_volume > 0:
            clearing_price = sum([p * q for _, _, q, p in all_matches]) / matched_volume
        else:
            clearing_price = (g_buy + g_sell) / 2.0
            
        # --- GRID INTERACTION (Residuals already updated in matching loop) ---
        grid_export = 0.0
        grid_import = 0.0
        
        # Unmatched Sellers -> Grid (MUST clear to maintain physical balance)
        for s in current_sell_orders:
            if s['qty'] > 1e-9:
                # Force settlement with grid at feed-in tariff
                trades[s['id']] += s['qty']
                grid_export += s['qty']
        
        # Unmatched Buyers -> Grid (MUST clear to maintain physical balance)
        for b in current_buy_orders:
            if b['qty'] > 1e-9:
                # Force settlement with grid at retail tariff
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
            raise ValueError(f"CRITICAL: Energy Conservation Violation. Net Trades: {net_trade_sum}, Grid Flow: {grid_flow}")
        
        info = {
            "total_volume": matched_volume,
            "clearing_price": clearing_price,
            "grid_flow": grid_flow
        }
        
        return trades, clearing_price, grid_flow, info
