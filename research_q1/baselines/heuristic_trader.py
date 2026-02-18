
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.data.data_loader import ResearchDataLoader

def run_heuristic_trader(
    date_str="2017-01-01", 
    data_path="processed_hybrid_data.csv",
    # Exact parameters from EnergyMarketEnvRecovery
    battery_capacities=[5.0, 5.0, 62.0, 10.0], 
    max_powers=[2.5, 2.5, 7.0, 5.0],
    efficiency=0.9
):
    print(f"\n Running Heuristic Trader for {date_str}...")
    
    # 1. Load Data
    loader = ResearchDataLoader(data_path)
    try:
        day_data = loader.get_test_day(date_str)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    demands, generations = loader.get_agent_data(day_data)
    prices_buy, prices_sell = loader.get_market_prices()
    
    n_agents = demands.shape[1]
    horizon = 24
    
    total_profit = 0.0
    
    # Initialize State
    socs = np.array(battery_capacities) * 0.5 # Start at 50%
    eff_sqrt = efficiency ** 0.5
    
    # Simulation Loop
    for t in range(horizon):
        # Current prices
        p_buy = prices_buy[t]
        p_sell = prices_sell[t]
        
        step_profit = 0.0
        
        for i in range(n_agents):
            # 1. Agent Physics Constraints
            base_cap = battery_capacities[i]
            p_max = max_powers[i]
            
            # Dynamic Cap (EV)
            current_cap = base_cap
            if i == 2: # Agent 2 is EV
                # Simplified hour assumption (t=hour)
                if 8 <= t < 17:
                    current_cap = 2.0
                    
            # Clip SoC to current capacity (e.g. if EV just left)
            socs[i] = min(socs[i], current_cap)
            
            # 2. Strategy: Greedy Self-Consumption
            # Net Load > 0: Deficit (Need power)
            # Net Load < 0: Surplus (Have power)
            net_load_pre_battery = demands[t, i] - generations[t, i]
            
            charge_kw = 0.0
            discharge_kw = 0.0
            grid_imp = 0.0
            grid_exp = 0.0
            
            if net_load_pre_battery < 0:
                # Excess Generation: Charge Battery first
                surplus = abs(net_load_pre_battery)
                
                # Max charge possible (Space / Rate)
                space_kwh = current_cap - socs[i]
                max_in = min(p_max, space_kwh / 1.0 / eff_sqrt) # dt=1h
                
                charge_kw = min(surplus, max_in)
                
                # Remaining goes to grid
                grid_exp = surplus - charge_kw
                
            else:
                # Deficit: Discharge Battery first
                deficit = net_load_pre_battery
                
                # Max discharge possible (Energy / Rate)
                max_out = min(p_max, socs[i] / 1.0 * eff_sqrt)
                
                discharge_kw = min(deficit, max_out)
                
                # Remaining comes from grid
                grid_imp = deficit - discharge_kw
                
            # 3. Update State
            # SoC[t+1] = SoC[t] + Charge*sqrt(eff) - Discharge/sqrt(eff)
            socs[i] += (charge_kw * eff_sqrt) - (discharge_kw / eff_sqrt)
            
            # 4. Calculate Profit
            # Revenue = Export * Price_Sell
            # Cost = Import * Price_Buy
            revenue = grid_exp * p_sell
            cost = grid_imp * p_buy
            
            step_profit += (revenue - cost)
            
        total_profit += step_profit
        
    print(f"Total System Profit: ${total_profit:.2f}")
    return {'total_profit': total_profit}

if __name__ == "__main__":
    run_heuristic_trader()
