import numpy as np

class DynamicMarketMechanism:
    """
    Dynamic, Incentive-Compatible P2P Market Mechanism.
    Adapts spread (delta) based on supply/demand imbalance.
    Includes realistic inefficiencies and stochastic price noise.
    """
    def __init__(self, 
                 grid_buy_price: float = 0.20,
                 grid_sell_price: float = 0.08,
                 base_delta: float = 0.03,
                 efficiency_range: tuple = (0.85, 0.95)):
        self.grid_buy = grid_buy_price
        self.grid_sell = grid_sell_price
        self.base_delta = base_delta
        self.efficiency_range = efficiency_range
        
        # Logging Hooks (accessible externally for RL training tracking)
        self.last_p2p_volume = 0.0
        self.last_prices = {"seller_price": 0.0, "buyer_price": 0.0}
        self.last_delta = 0.0

    def update_prices(self, grid_buy: float, grid_sell: float) -> None:
        """Update reference grid prices from external environmental signal."""
        self.grid_buy = grid_buy
        self.grid_sell = grid_sell

    def compute_delta(self, total_supply: float, total_demand: float) -> float:
        """Dynamic delta computation based on market imbalance."""
        # Calculate imbalance bounded between -1 and +1
        imbalance = (total_supply - total_demand) / (total_supply + total_demand + 1e-6)
        
        # Adjust base spread dynamically
        delta = self.base_delta * (1.0 + imbalance)
        
        # Clamp to realistic defined bounds
        delta = np.clip(delta, 0.01, 0.08)
        
        # Secondary safety clamp against inverted market structures
        max_delta = (self.grid_buy - self.grid_sell) / 2.0 - 0.001
        return min(delta, max_delta)

    def compute_prices(self, delta: float) -> dict:
        """Compute fair-value prices utilizing stochastic noise."""
        noise_s = np.random.normal(0, 0.01)
        noise_b = np.random.normal(0, 0.01)
        
        seller_price = self.grid_sell + delta + noise_s
        buyer_price = self.grid_buy - delta + noise_b
        
        # Strict bound clamping against boundary failures
        seller_price = np.clip(seller_price, self.grid_sell + 0.001, self.grid_buy - 0.001)
        buyer_price = np.clip(buyer_price, self.grid_sell + 0.001, self.grid_buy - 0.001)
        
        # Prevent the exchange from eating negative spreads (market maker invariant)
        if seller_price > buyer_price:
            mid = (seller_price + buyer_price) / 2.0
            seller_price = mid - 0.001
            buyer_price = mid + 0.001
            
        return {
            "seller_price": float(seller_price),
            "buyer_price": float(buyer_price)
        }

    def match_trades(self, orders: np.ndarray) -> dict:
        """
        Determines the total matched volume with integrated market inefficiency.
        """
        buyers_mask = orders > 0
        sellers_mask = orders < 0
        
        demand = np.where(buyers_mask, orders, 0.0)
        supply = np.where(sellers_mask, np.abs(orders), 0.0)
        
        total_demand = np.sum(demand)
        total_supply = np.sum(supply)
        
        # Imperfect matching algorithm introduces realistic fractional efficiency
        efficiency = np.random.uniform(self.efficiency_range[0], self.efficiency_range[1])
        traded_energy = efficiency * min(total_supply, total_demand)
        
        # Calculate dynamic state variables
        delta = self.compute_delta(total_supply, total_demand)
        imbalance = (total_supply - total_demand) / (total_supply + total_demand + 1e-6)
        
        return {
            "demand": demand,
            "supply": supply,
            "total_demand": total_demand,
            "total_supply": total_supply,
            "traded_energy": traded_energy,
            "delta": delta,
            "imbalance": imbalance,
            "efficiency": efficiency,
            "buyers_mask": buyers_mask,
            "sellers_mask": sellers_mask
        }

    def allocate_energy(self, match_data: dict, orders: np.ndarray) -> dict:
        """Proportional allocation algorithm over pooled resources."""
        total_demand = match_data["total_demand"]
        total_supply = match_data["total_supply"]
        traded_energy = match_data["traded_energy"]
        demand = match_data["demand"]
        supply = match_data["supply"]
        
        buy_ratio = (traded_energy / total_demand) if total_demand > 0 else 0.0
        sell_ratio = (traded_energy / total_supply) if total_supply > 0 else 0.0
        
        p2p_trades = np.zeros_like(orders)
        
        p2p_trades[match_data["buyers_mask"]] = demand[match_data["buyers_mask"]] * buy_ratio
        p2p_trades[match_data["sellers_mask"]] = - (supply[match_data["sellers_mask"]] * sell_ratio)
        
        unmatched_demand = demand - np.where(p2p_trades > 0, p2p_trades, 0.0)
        unmatched_supply = supply - np.where(p2p_trades < 0, np.abs(p2p_trades), 0.0)
        
        return {
            "p2p_trades": p2p_trades,
            "unmatched_demand": unmatched_demand,
            "unmatched_supply": unmatched_supply
        }
        
    def settle_market(self, p2p_trades: np.ndarray, unmatched_demand: np.ndarray, 
                      unmatched_supply: np.ndarray, prices: dict) -> dict:
        """Determines settlements per agent factoring intra-system trades vs external grid."""
        # 1. P2P Settlement Process
        seller_revenue = np.zeros_like(p2p_trades)
        buyer_cost = np.zeros_like(p2p_trades)
        
        sellers_mask = p2p_trades < 0
        buyers_mask = p2p_trades > 0
        
        seller_revenue[sellers_mask] = np.abs(p2p_trades[sellers_mask]) * prices["seller_price"]
        buyer_cost[buyers_mask] = p2p_trades[buyers_mask] * prices["buyer_price"]
        
        # 2. Grid Interactivity 
        grid_import = unmatched_demand
        grid_export = unmatched_supply
        
        return {
            "seller_revenue": seller_revenue,
            "buyer_cost": buyer_cost,
            "grid_import": grid_import,
            "grid_export": grid_export
        }

    def clear_market(self, orders: list) -> dict:
        """Main execution flow orchestrating end-to-end clearing for the timestep."""
        orders_arr = np.array(orders, dtype=np.float64)
        
        # 1. Matching Logic
        match_data = self.match_trades(orders_arr)
        
        # 2. Pricing Execution
        prices = self.compute_prices(match_data["delta"])
        
        # 3. Liquidity Allocation
        alloc_data = self.allocate_energy(match_data, orders_arr)
        
        # 4. Final Settlement Array
        settlement = self.settle_market(
            alloc_data["p2p_trades"], 
            alloc_data["unmatched_demand"], 
            alloc_data["unmatched_supply"],
            prices
        )
        
        # 5. Energy Conservation Invariant
        net_order = np.sum(orders_arr)
        net_grid = np.sum(settlement["grid_import"]) - np.sum(settlement["grid_export"])
        if not np.isclose(net_order, net_grid, atol=1e-5):
            raise ValueError(f"Global energy conservation invariant failed! Net mismatch: {net_grid - net_order}")
            
        p2p_volume = match_data["traded_energy"]
        
        # 6. RL State Hooking Log Update
        self.last_p2p_volume = p2p_volume
        self.last_prices = prices
        self.last_delta = match_data["delta"]
        
        # Construct exact metric payload output
        output = {
            "p2p_trades": alloc_data["p2p_trades"].tolist(),
            "seller_revenue": settlement["seller_revenue"].tolist(),
            "buyer_cost": settlement["buyer_cost"].tolist(),
            "grid_import": settlement["grid_import"].tolist(),
            "grid_export": settlement["grid_export"].tolist(),
            "clearing_prices": prices,
            "p2p_volume": float(p2p_volume),
            "seller_price": prices["seller_price"],
            "buyer_price": prices["buyer_price"],
            "delta": float(match_data["delta"]),
            "imbalance": float(match_data["imbalance"]),
            "efficiency": float(match_data["efficiency"])
        }
        
        return output

def test_dynamic_market():
    """Validates the DynamicMarketMechanism implementation across distribution structures."""
    print("--- Running Dynamic Market Validation ---")
    np.random.seed(42)  # Maintain deterministic outcomes for the test outputs
    
    market = DynamicMarketMechanism()
    test_cases = [
        ("Balanced Distribution", [10.0, -10.0, 5.0, -5.0]),
        ("Supply Dominant Distribution", [5.0, -15.0, 2.0, -8.0]),
        ("Demand Dominant Distribution", [20.0, -5.0, 15.0, -2.0])
    ]
    
    for name, orders in test_cases:
        print(f"\\nTest Case => {name}")
        print(f"Input Orders: {orders}")
        results = market.clear_market(orders)
        
        print(f"  P2P Volume:      {results['p2p_volume']:.2f} kWh (Efficiency: {results['efficiency']:.2f})")
        print(f"  Seller Price:    ${results['seller_price']:.4f}/kWh")
        print(f"  Buyer Price:     ${results['buyer_price']:.4f}/kWh")
        print(f"  Grid Import:     {np.sum(results['grid_import']):.2f} kWh")
        print(f"  Grid Export:     {np.sum(results['grid_export']):.2f} kWh")
        print(f"  Delta Spread:    {results['delta']:.4f} (Imbalance: {results['imbalance']:.2f})")
        
        # Enforce non-negative bounds constraints invariant
        assert np.all(np.array(results["grid_import"]) >= 0)
        assert np.all(np.array(results["grid_export"]) >= 0)
    
    print("\n[VALID] All bounds, logic routing constraints, and energy conservations check OK!")

if __name__ == "__main__":
    test_dynamic_market()
