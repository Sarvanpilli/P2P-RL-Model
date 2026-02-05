# P2P-RL-Model: Project Overview & Implementation Guide

## 1. Project Description
This project implements a **Multi-Agent Reinforcement Learning (MARL)** system for **Peer-to-Peer (P2P) Energy Trading**. The goal is to create an autonomous "Perfect Agent" that manages distributed energy resources (Battery, Solar PV) to maximize economic profit, minimize CO2 emissions, and ensure grid stability.

The system simulates a microgrid where prosumers (households with solar+battery) can trade energy with each other (P2P) or with the main grid, using a dynamic pricing mechanism learned by the AI.

## 2. Key Improvements & Features
We have evolved the project from a basic baseline to a robust, research-ready system.

### A. Advanced Market Logic (Limit Orders)
*   **Old**: Simple quantity-based matching.
*   **New**: **Uniform Price Double Auction**. Agents submit **Limit Orders** (Quantity + Price Bid).
*   **Mechanism**: The market engine sorts Buyers (High to Low) and Sellers (Low to High) to find a single **Clearing Price** where Supply meets Demand. This mimics real-world electricity markets.

### B. Robust Safety Layer (Feasibility Filter)
*   **Problem**: RL agents often output physically impossible actions (e.g., discharging an empty battery).
*   **Solution**: Implemented a **Deterministic, Stateless Feasibility Filter**.
    *   **Physics**: Clips battery actions to exact SoC limits and power ratings.
    *   **Energy Balance**: Ensures agents only trade what they physically have (Surplus) or need (Deficit).
    *   **Idempotence**: Verified that applying the filter multiple times does not change the result (stability).

### C. Environment Enhancements
*   **CO2 Tracking**: The grid now has a dynamic CO2 intensity. Agents are penalized for importing dirty energy.
*   **Forecast Uncertainty**: Observations include noisy forecasts for PV and Demand, forcing the agent to be robust against prediction errors.
*   **Explicit Physics**: Power flow calculations now strictly use `dt_hours` to ensure Energy (kWh) = Power (kW) × Time (h) is conserved.

### D. Validation & Metrics
*   **Overfit Test**: Verified that the agent can learn a perfect policy in a deterministic environment.
*   **Stress Tests**: Verified system stability under "Extreme Shortage" (Winter Night) and "Extreme Surplus" (Summer Day) scenarios.
*   **Reporting**: Automated generation of `evaluation_results.csv` and visualization plots (`market_prices.png`, `grid_flow.png`).

## 3. Implementation Workflow

The project was implemented in 8 distinct phases:

1.  **Environment Setup**: Created the `EnergyMarketEnv` with basic battery and grid physics.
2.  **Baseline RL**: Implemented PPO (Proximal Policy Optimization) using Stable Baselines3.
3.  **Real-Time Wrapper**: Built a dashboard to visualize agent actions on realistic daily profiles.
4.  **Action Space Expansion**: Added **Price Bidding** to the agent's capabilities.
5.  **Market Engine Upgrade**: Replaced simple matching with the **Limit Order Book** logic.
6.  **Safety Integration**: Built and integrated the `FeasibilityFilter`.
7.  **Robustness Overhaul**: Added unit tests for energy conservation, filter idempotence, and deterministic learning.
8.  **Final Verification**: Retrained the model and ran full-day evaluations to confirm performance.

## 4. How to Run the Project

### A. Train the Agent
Train a new model from scratch (approx. 10-20 mins for 100k steps).
```bash
python train/train_sb3_ppo.py --timesteps 100000 --seed 42
```

### B. Verify Robustness
Run the suite of unit tests to ensure physics and logic are sound.
```bash
python tests/test_robustness.py
```

### C. Evaluate Performance
Run a 24-hour simulation using the trained model and generate plots.
```bash
python evaluation/evaluate_episode.py
python evaluation/plot_results.py
```
*Outputs*: Check the `evaluation/` folder for CSVs and PNGs.

### D. Real-Time Dashboard
Watch the agents trade in "real-time" (accelerated).
```bash
python run_realtime.py
```

## 5. Final Status
| Component | Status | Notes |
| :--- | :--- | :--- |
| **Physics Engine** | ✅ **Robust** | Energy conserved, efficiency applied correctly. |
| **Market Mechanism** | ✅ **Advanced** | Uniform Price Auction with Limit Orders. |
| **Safety** | ✅ **Verified** | Filter prevents all illegal actions. |
| **AI Agent** | ✅ **Trained** | Learns profitable and safe strategies. |
| **Code Quality** | ✅ **High** | Modular, tested, and reproducible. |
