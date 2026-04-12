"""
Create a heterogeneous prosumer dataset from existing local sources.

Inputs (already in repo):
- evaluation/ausgrid_p2p_energy_dataset.csv : hourly PV + demand (4 agents, 2017)
- evaluation/wind_generation_data.csv       : hourly wind power (kW), 2017

Outputs:
- evaluation/hybrid_public_dataset.csv : hourly data with four archetypes:
    agent_0_pv_kw, agent_0_demand_kw   (solar home, original Ausgrid)
    agent_1_wind_kw, agent_1_demand_kw (wind home, demand from agent_1, wind from wind file)
    agent_2_pv_kw, agent_2_demand_kw   (EV/V2G home; PV from agent_2, demand = orig + EV load)
    agent_3_pv_kw, agent_3_demand_kw   (standard load; PV from agent_3, demand from agent_3)

Notes:
- EV charging synthetic profile: adds 3.6 kW from 18:00–21:00 local time daily (typical L2 home charger).
- All time columns preserved as 'hour' index 0..8759.
"""

import os
import pandas as pd
import numpy as np

AUSGRID_PATH = os.path.join("evaluation", "ausgrid_p2p_energy_dataset.csv")
WIND_PATH = os.path.join("evaluation", "wind_generation_data.csv")
OUT_PATH = os.path.join("evaluation", "hybrid_public_dataset.csv")


def load_sources():
    pv_load = pd.read_csv(AUSGRID_PATH)
    wind = pd.read_csv(WIND_PATH)
    return pv_load, wind


def make_ev_load(hours: int) -> np.ndarray:
    """Simple evening EV charge: 3.6 kW for 3 hours daily."""
    ev = np.zeros(hours)
    for h in range(hours):
        if 18 <= (h % 24) < 21:
            ev[h] = 3.6
    return ev


def main():
    pv_load, wind = load_sources()
    hours = len(pv_load)

    ev_extra = make_ev_load(hours)

    out = pd.DataFrame()
    out["hour"] = pv_load["hour"]

    # Agent 0: solar baseline (unchanged)
    out["agent_0_pv_kw"] = pv_load["agent_0_pv_kw"]
    out["agent_0_demand_kw"] = pv_load["agent_0_demand_kw"]

    # Agent 1: wind + demand from agent_1
    # Wind file has column 'Power' in kW
    out["agent_1_wind_kw"] = wind["Power"]
    out["agent_1_demand_kw"] = pv_load["agent_1_demand_kw"]

    # Agent 2: EV home with PV and extra EV load
    out["agent_2_pv_kw"] = pv_load["agent_2_pv_kw"]
    out["agent_2_demand_kw"] = pv_load["agent_2_demand_kw"] + ev_extra

    # Agent 3: standard home with small PV
    out["agent_3_pv_kw"] = pv_load["agent_3_pv_kw"]
    out["agent_3_demand_kw"] = pv_load["agent_3_demand_kw"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH} (rows={len(out)})")


if __name__ == "__main__":
    main()
