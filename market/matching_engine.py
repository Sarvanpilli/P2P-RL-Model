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

    def __init__(self, base_price=0.2):
        self.base_price = base_price

    def match(self, orders):
        orders = np.array(orders, dtype=float)
        sellers_mask = orders > 0
        buyers_mask = orders < 0
        total_sell = float(np.sum(orders[sellers_mask]))  # positive kW
        total_buy = float(-np.sum(orders[buyers_mask]))  # positive kW

        # Determine matched volume
        matched = min(total_sell, total_buy)
        # allocate matched amounts proportionally
        trades = np.zeros_like(orders)
        if matched > 0:
            if total_sell > 0:
                trades[sellers_mask] += orders[sellers_mask] * (matched / total_sell)
            if total_buy > 0:
                trades[buyers_mask] += -np.abs(orders[buyers_mask]) * (matched / total_buy)
                trades[buyers_mask] *= -1  # keep sign negative for buys

        # grid_import = remaining demand (if any) that grid must supply; positive -> grid supplies
        grid_import = max(0.0, total_buy - matched) - max(0.0, total_sell - matched)  # kW

        # clearing price: base + factor * imbalance ratio
        imbalance = total_buy - total_sell
        price = max(0.01, self.base_price * (1.0 + 0.1 * (imbalance)))
        # small heuristic: if high demand, price rises; if surplus, price lowers slightly
        return trades, price, grid_import
