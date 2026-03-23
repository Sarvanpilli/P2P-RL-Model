# dashboard/data_results.py

seed_results = [
    {"seed": 0,  "p2p": 1004.23, "reward": 139.85, "buyers": 1.851},
    {"seed": 7,  "p2p":  937.02, "reward": 137.69, "buyers": 1.810},
    {"seed": 13, "p2p":  966.46, "reward": 136.99, "buyers": 1.762},
    {"seed": 21, "p2p": 1120.04, "reward": 132.68, "buyers": 1.911},
    {"seed": 42, "p2p":  989.68, "reward": 146.99, "buyers": 1.756},
]

comparative_results = {
    "baseline":      {"p2p": 0.00,   "p2p_std": 0.00,  "buyers": 0.00},
    "legacy_auction": {"p2p": 67.83,  "p2p_std": 27.29, "buyers": 0.00},
    "slim_v2":        {"p2p": 992.77, "p2p_std": 60.47, "buyers": 1.777, "buyers_std": 0.049, "reward": 133.89, "reward_std": 5.92, "violations": 0},
}

scalability_data = [
    {"n": 4,  "p2p_per_agent": 63.66, "profit_per_agent": -38.05, "change": None},
    {"n": 6,  "p2p_per_agent": 71.12, "profit_per_agent": -36.42, "change": '+11.7%'},
    {"n": 8,  "p2p_per_agent": 82.45, "profit_per_agent": -35.11, "change": '+29.5%'},
    {"n": 10, "p2p_per_agent": 94.20, "profit_per_agent": -33.88, "change": '+48.0%'},
]

convergence_data = [
    {"step": 50,  "reward": -12.37, "phase": 'Phase 5'},
    {"step": 100, "reward": -10.41, "phase": 'Phase 5'},
    {"step": 150, "reward": -6.91,  "phase": 'Phase 5'},
    {"step": 200, "reward": -7.89,  "phase": 'Phase 5'},
    {"step": 250, "reward": -12.85, "phase": 'Phase 5'},
    {"step": 300, "reward": 133.89, "phase": 'SLIM v2'},
]

agents = [
    {"id": 0, "name": 'Solar prosumer',  "icon": '☀️', "color": '#EF9F27', "battery": 5,  "rate": 2.5, "source": 'Solar PV'},
    {"id": 1, "name": 'Wind prosumer',   "icon": '🌬️', "color": '#378ADD', "battery": 5,  "rate": 2.5, "source": 'Wind turbine'},
    {"id": 2, "name": 'EV/V2G',          "icon": '🚗', "color": '#1D9E75', "battery": 62, "rate": 7.0, "source": 'Home PV + V2G'},
    {"id": 3, "name": 'Standard',        "icon": '🏠', "color": '#888780', "battery": 10, "rate": 5.0, "source": 'Solar PV'},
]

# Timeline phases
phases = [
    "Environment Setup",
    "Real Data Integration",
    "Grid-Aware Safety",
    "LSTM + Forecasting",
    "Heterogeneous Agents",
    "Nash Fix + Curriculum",
    "SLIM v2 Final Evaluation",
    "GNN Attention Analysis"
]
