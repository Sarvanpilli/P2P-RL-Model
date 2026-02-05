import numpy as np
import pandas as pd

def generate_profile(n_agents=4, hours=24):
    # Time index
    t = np.arange(hours)
    
    data = {"hour": t}
    
    for i in range(n_agents):
        # 1. Solar PV: Bell curve centered at noon (hour 12)
        # Peak around 10-20 kW
        peak_pv = np.random.uniform(10, 20)
        # Gaussian: exp(- (x - mu)^2 / (2 * sigma^2))
        pv_curve = peak_pv * np.exp(- (t - 12)**2 / (2 * 3.0**2))
        # Add some noise
        pv_curve += np.random.normal(0, 0.5, size=hours)
        pv_curve = np.clip(pv_curve, 0, None)
        
        # 2. Demand: Two peaks (Morning 8-10, Evening 18-21)
        base_load = np.random.uniform(1, 3)
        morning_peak = np.random.uniform(5, 10) * np.exp(- (t - 9)**2 / (2 * 1.5**2))
        evening_peak = np.random.uniform(8, 15) * np.exp(- (t - 19)**2 / (2 * 2.0**2))
        
        demand_curve = base_load + morning_peak + evening_peak
        # Add noise
        demand_curve += np.random.normal(0, 0.5, size=hours)
        demand_curve = np.clip(demand_curve, 0, None)
        
        data[f"agent_{i}_pv_kw"] = pv_curve
        data[f"agent_{i}_demand_kw"] = demand_curve

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_profile()
    df.to_csv("test_day_profile.csv", index=False)
    print("Generated 'test_day_profile.csv' with realistic 24h data.")
    print(df.head())
