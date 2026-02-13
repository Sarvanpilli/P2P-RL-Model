
import numpy as np

class RuleBasedAgent:
    """
    A 'Truthful' or 'Zero-Intelligence' constrained agent for Double Auction baselines.
    
    Strategy:
    1. Battery: 
       - If Price is Low (Off-Peak): Charge.
       - If Price is High (Peak): Discharge.
       - Else: Idle or Self-Consume.
    2. Bidding (Truthful-ish):
       - Buy Limit: Willing to pay up to Retail Price (Grid equivalent).
       - Sell Limit: Willing to sell down to Feed-in Tariff (Grid equivalent).
       - Quantity: Net Surplus/Deficit after battery action.
    """
    
    def __init__(self, agent_id, battery_capacity_kwh, max_rate_kw):
        self.agent_id = agent_id
        self.battery_capacity_kwh = battery_capacity_kwh
        self.max_rate_kw = max_rate_kw
        
    def get_action(self, obs, time_step_hour):
        """
        Returns action: [Battery_kW, Grid_Trade_kW, Price_Bid]
        
        Args:
            obs: Flattened observation (we might need to parse it or just use internal state if we had it).
                 Since this is a baseline, we'll assume we can cheat/access state or parse obs.
                 Obs structure: [Dem, SoC, PV, ..., Retail, FeedIn, ...]
        """
        # Parse Observation (Assuming standard robust env structure)
        # Base dim 8: [Dem, SoC, PV, Export, Import, CO2, Retail, FeedIn]
        demand = obs[0]
        soc = obs[1]
        pv = obs[2]
        retail = obs[6]
        feed_in = obs[7]
        
        # 1. Determine Battery Action (Simple Arbitrage)
        # Heuristic: 
        # - Charge if Retail is low (Off-peak) AND SoC < 90%
        # - Discharge if Retail is high (Peak) AND SoC > 10%
        # - Else: Support local load (Self-Consumption)
        
        batt_action = 0.0
        
        # Simple ToU Logic based on price levels
        # Assuming we know typical spread. 
        # Low < 0.15, High > 0.30
        
        is_low_price = (retail < 0.15)
        is_high_price = (retail > 0.30)
        
        if is_low_price and soc < 0.9 * self.battery_capacity_kwh:
            # Charge max
            batt_action = self.max_rate_kw
        elif is_high_price and soc > 0.1 * self.battery_capacity_kwh:
            # Discharge max
            batt_action = -self.max_rate_kw
        else:
            # Self-Consumption / Balancing
            # If PV > Demand, Charge surplus
            # If PV < Demand, Discharge deficit
            net_load = demand - pv
            if net_load < 0: # Surplus
                # Charge surplus
                current_surplus = abs(net_load)
                batt_action = min(current_surplus, self.max_rate_kw)
            else: # Deficit
                # Discharge to meet load
                current_deficit = net_load
                batt_action = -min(current_deficit, self.max_rate_kw)

        # 2. Determine Net Trade
        # Net = PV - Demand - Battery_Flow (+Charge, -Discharge)
        # If Net > 0: Surplus to Sell
        # If Net < 0: Deficit to Buy
        
        # Note: We must respect SoC logic, but Env handles physics.
        # We output desired battery action.
        
        projected_net_gen = pv - demand - batt_action
        
        trade_kw = 0.0
        price_bid = 0.0
        
        if projected_net_gen > 0:
            # Selling
            trade_kw = projected_net_gen # Positive
            # Strategy: Sell if price > Feed-in.
            # Truthful bid: Marginal cost. Solar is free. Battery degradation is cost.
            # Let's simple bid: Feed-in + epsilon (to prefer P2P over Grid)
            price_bid = feed_in + 0.01 
        else:
            # Buying
            trade_kw = projected_net_gen # Negative
            # Strategy: Buy if price < Retail.
            # Truthful bid: Retail - epsilon (to prefer P2P over Grid)
            price_bid = retail - 0.01
            
        return np.array([batt_action, trade_kw, price_bid], dtype=np.float32)

