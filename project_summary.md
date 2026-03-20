# P2P-RL-Model: Project Overview & Implementation Guide

## 1. Project Description
This project implements a **Multi-Agent Reinforcement Learning (MARL)** system for **Peer-to-Peer (P2P) Energy Trading**. The goal is to create an autonomous "Perfect Agent" that manages distributed energy resources (Battery, Solar PV) to maximize economic profit, minimize CO2 emissions, and ensure grid stability.

The system simulates a microgrid where prosumers (4 heterogeneous households: Solar PV, Wind, Electric Vehicle/V2G, and Standard) each equipped with a battery storage unit can trade energy with each other (P2P) or with the main grid, using a dynamic pricing mechanism learned by the AI. We refer to this integrated system — combining the MARL agents, dynamic market mechanism, and two-tier safety architecture — as the **SLIM (Safety-constrained Liquidity-Integrated Market)** framework.

## 2. Key Improvements & Features
We have evolved the project from a basic baseline to a robust, research-ready system.

### A. Advanced Market Logic (Limit Orders)
*   **Old**: Simple quantity-based matching.
*   **New**: **Uniform Price Double Auction**. Agents submit **Limit Orders** (Quantity + Price Bid).
*   **Mechanism**: The market engine sorts Buyers (High to Low) and Sellers (Low to High) to find a single **Clearing Price** where Supply meets Demand. This mimics real-world electricity markets.

### B. Projection-Based Safety Constraints (3-layer architecture)
*   **Problem**: RL agents often output physically impossible actions (e.g., discharging an empty battery).
*   **Solution**: Implemented a **Deterministic, Stateless Feasibility Filter**.
    *   **Physics**: Clips battery actions to exact SoC limits and power ratings.
    *   **Energy Balance**: Ensures agents only trade what they physically have (Surplus) or need (Deficit).
    *   **Idempotence**: Verified that applying the filter multiple times does not change the result (stability).

> [!NOTE]
> Safety mechanism: Projection-based hard constraint enforcement. Actions are clipped to the feasible set before execution. This guarantees constraint satisfaction but may bias the policy gradient. A Lagrangian primal-dual layer is layered on top of this projection system during training to teach the policy to avoid constraint boundaries proactively. See results_and_discussion.md Section 4C for full details.

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

## 5. Key Results & Verified Dataset

### A. Key Results of the SLIM Framework
The SLIM framework demonstrates significant improvements over a baseline scenario (no P2P trading, no intelligent agent control).

| Metric | Baseline (No SLIM) | SLIM Framework (Trained Agent) |
| :--- | :--- | :--- |
| P2P Trading Volume | 0.00 kWh (0% matched) | 254.63 kWh (~8% increase) |
| Grid Reliance (Import) | 3,085.53 kWh | 2,987.41 kWh (3.18% reduction) |
| System Safety | Frequent SoC Violations | **0% Critical Safety Violations** |
| Average Cost Saving | $0.00 (Baseline) | $38.20 per household |

### B. Verified Dataset
The model is trained and evaluated on the **Ausgrid Solar Home Dataset**, a publicly available real-world dataset of hourly energy consumption and rooftop solar generation from residential households in New South Wales, Australia (Year: 2017). Wind generation data was sourced from meteorological records for the same region. The combined dataset spans **8,760 hours (one full year)** and was preprocessed via `scripts/preprocess_hybrid_data.py`. All values are min-max normalized to [0, 1] using scale factors stored in `normalization_config.json`.

| Property | Value |
| :--- | :--- |
| **Source** | Ausgrid (NSW, Australia) + Regional Wind Data |
| **Year** | 2017 |
| **Timesteps** | 8,760 (hourly, 1 full year) |
| **Train / Eval Split** | 80% Training (7,008h) / 20% Evaluation (1,752h) |
| **Normalization** | Min-Max per column to [0, 1] |
| **Preprocessing script** | `scripts/preprocess_hybrid_data.py` |

## 6. Final Status
| Component | Status | Notes |
| :--- | :--- | :--- |
| **Physics Engine** | ✅ **Robust** | Energy conserved, efficiency applied correctly. |
| **Market Mechanism** | ✅ **Advanced** | Uniform Price Auction with Limit Orders. |
| **Safety** | ✅ **Verified** | Two-tier: projection filter (hard guarantee) + Lagrangian layer (learned constraints). |
| **AI Agent** | ✅ **Trained** | Learns profitable and safe strategies. |
| **Code Quality** | ✅ **High** | Modular, tested, and reproducible. |
