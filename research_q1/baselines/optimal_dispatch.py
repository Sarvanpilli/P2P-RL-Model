
import pulp
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.data.data_loader import ResearchDataLoader

def solve_optimal_dispatch(
    date_str="2017-01-01", 
    data_path="processed_hybrid_data.csv",
    # Exact parameters from EnergyMarketEnvRecovery
    battery_capacities=[5.0, 5.0, 62.0, 10.0], 
    max_powers=[2.5, 2.5, 7.0, 5.0],
    efficiency=0.9
):
    print(f"\n Solving Optimal Dispatch for {date_str}...")
    print("Agent Parameters:")
    for i in range(len(battery_capacities)):
        print(f"  Agent {i}: Cap={battery_capacities[i]}kWh, MaxP={max_powers[i]}kW")
        
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
    
    # 2. Setup Optimization Problem
    prob = pulp.LpProblem("Microgrid_Optimal_Dispatch", pulp.LpMaximize)
    
    # Decision Variables
    # shape: (n_agents, horizon)
    grid_import = pulp.LpVariable.dicts("Import", ((i, t) for i in range(n_agents) for t in range(horizon)), lowBound=0)
    grid_export = pulp.LpVariable.dicts("Export", ((i, t) for i in range(n_agents) for t in range(horizon)), lowBound=0)
    charge = pulp.LpVariable.dicts("Charge", ((i, t) for i in range(n_agents) for t in range(horizon)), lowBound=0)
    discharge = pulp.LpVariable.dicts("Discharge", ((i, t) for i in range(n_agents) for t in range(horizon)), lowBound=0)
    # Note on SoC upper bound: we set it per-step for dynamic constraints
    soc = pulp.LpVariable.dicts("SoC", ((i, t) for i in range(n_agents) for t in range(horizon+1)), lowBound=0)
    
    # Objective Function: Maximize Profit (Export Revenue - Import Cost)
    objective = []
    for i in range(n_agents):
        for t in range(horizon):
            revenue = grid_export[i, t] * prices_sell[t]
            cost = grid_import[i, t] * prices_buy[t]
            objective.append(revenue - cost)
            
    prob += pulp.lpSum(objective)
    
    # Constraints
    for i in range(n_agents):
        base_cap = battery_capacities[i]
        p_max = max_powers[i]
        
        # Initial SoC
        prob += soc[i, 0] == base_cap * 0.5 
        
        for t in range(horizon):
            # 1. Dynamic Capacity (EV Constraint)
            # Agent 2 is EV (0-indexed -> 2)
            # Logic: 8 <= hour < 17 -> Cap = 2.0
            current_cap = base_cap
            if i == 2: # Agent 2 is EV
                hour = t # simplified, assuming data starts at 00:00
                if 8 <= hour < 17:
                    current_cap = 2.0
            
            prob += soc[i, t] <= current_cap
            
            # 2. Power limits
            prob += charge[i, t] <= p_max
            prob += discharge[i, t] <= p_max
            
            # 3. Power Balance
            # Gen + Import + Discharge = Demand + Export + Charge
            prob += (generations[t, i] + grid_import[i, t] + discharge[i, t] == 
                     demands[t, i] + grid_export[i, t] + charge[i, t])
            
            # 4. Battery Dynamics
            prob += soc[i, t+1] == soc[i, t] + (charge[i, t] * efficiency) - (discharge[i, t] / efficiency)
            
        # Final SoC check
        prob += soc[i, horizon] <= base_cap # Should be valid
    
    # 3. Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    
    if status != "Optimal":
        print(f"Optimization failed: {status}")
        return None
        
    profit = pulp.value(prob.objective)
    print(f"Total System Profit: ${profit:.2f}")
    
    return {'total_profit': profit}

if __name__ == "__main__":
    try:
        import pulp
    except ImportError:
        print("Error: 'pulp' library not found. Please install it using: pip install pulp")
        sys.exit(1)
        
    solve_optimal_dispatch()
