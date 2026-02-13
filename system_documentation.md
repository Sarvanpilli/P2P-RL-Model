
# P2P-RL Grid-Aware Energy Trading System
**Comprehensive System Documentation**

## 1. Project Overview
This project implements a research-grade **Multi-Agent Reinforcement Learning (MARL)** framework for **Peer-to-Peer (P2P) Energy Trading**. The system is designed to optimize energy management for prosumers (households with Solar PV + Battery) connected in a microgrid.

**Core Objective**: Transform "dumb" prosumers into **Grid-Aware Intelligent Agents** that:
1.  Maximize Economic Profit (Trading P2P > Exporting to Grid).
2.  Minimize Grid Dependence (Self-consumption + P2P > Importing from Grid).
3.  Respect Physical Constraints (Battery health, Grid limits).

---

## 2. System Architecture

The architecture follows a Layered Control approach:

### Layer 1: The Environment (`train/energy_env_robust.py`)
A `Gymnasium`-based environment simulating the microgrid physics and market.
*   **Physics Engine**:
    *   **Battery Dynamics**: Tracks SoC (kWh), enforces robust Ramp Rate limits.
    *   **Grid Physics**: Calculates Power Flow and Distribution Losses ($I^2R$) based on line resistance.
    *   **Real Data**: Powered by **Ausgrid** Solar Home Electricity Data.
*   **Market Engine**:
    *   Type: **Double Auction** with Limit Orders.
    *   Mechanism: Matches internal Buy/Sell orders first (P2P) before settling with the main Grid.
    *   Fairness: Uses a Gini-coefficient-based penalty in the reward (optional).

### Layer 2: The Agent (RL Policy)
*   **Algorithm**: **PPO (Proximal Policy Optimization)** from `stable-baselines3`.
*   **Observation Space**:
    *   Demand (kW), PV Gen (kW), SoC (%), Grid Prices ($/kWh), Cumulative Imports/Exports.
*   **Action Space**:
    *   `battery_action` (Charge/Discharge kW).
    *   `grid_action` (Import/Export kW request).
    *   `price_bid` (Willingness to pay/accept).

### Layer 3: Reward System (`train/reward_tracker.py`)
A custom, modular reward function refactored to be "Grid-Aware".
*   **Formula**:
    $$ R = w_{profit} \cdot \text{Profit} - w_{grid} \cdot \text{GridImport} - w_{soc} \cdot \text{SoCPenalty} - w_{degrad} \cdot \text{Throughput} $$
*   **Key Shift**: Unlike standard profit-maximizers, this system explicitly penalizes Grid Imports ($w_{grid}=0.5$) to encourage self-sufficiency and local trading, serving as a proxy for carbon footprint reduction.

### Layer 4: Safety & Robustness (`train/autonomous_guard.py`)
*   **Autonomous Guard**: Acts as a "Safety Supervisor" wrapping the logic.
*   **Feasibility Filter**: Intercepts RL actions to ensure they are physically possible (e.g., preventing discharge of an empty battery).

---

## 3. Codebase Structure

### A. Core Training (`train/`)
*   `energy_env_robust.py`: The main environment file. **(Critical)**
*   `reward_tracker.py`: Handles reward component calculation and logging. **(Critical)**
*   `train_phase3_grid_aware.py`: The active training script for the Grid-Aware configuration.
*   `autonomous_guard.py`: Safety filtering logic.

### B. Evaluation (`evaluation/`)
*   `evaluate_phase3.py`: Comparison script (RL Phase 2 vs RL Phase 3 vs Baseline).
*   `evaluate_real.py`: Long-horizon evaluation on Ausgrid data.
*   `plot_phase3.py`: Generates the comparison plots.
*   `ausgrid_p2p_energy_dataset.csv`: Real-world input data.
*   `results_*.csv`: Output logs from evaluations.

### C. Baselines (`baselines/`)
*   `rule_based_agent.py`: A truthful-bidding heuristic agent used for benchmarking.

### D. Legacy (`unused/`)
*   Contains deprecated scripts (`train_phase2_advanced.py`, `dashboard.py`, etc.) moved here during cleanup to keep the workspace focused.

---

## 4. Usage Guide

### Prerequisites
```bash
pip install gymnasium stable-baselines3 pandas numpy matplotlib seaborn
```

### 1. Training the Agent (Phase 3)
To train the Grid-Aware PPO agent:
```bash
python train/train_phase3_grid_aware.py
```
*   **Output**: Saves model to `models_phase3/ppo_grid_aware.zip`.
*   **Logs**: Tensorboard logs in `tensorboard_logs/`.

### 2. Evaluating Performance
To run a comparative evaluation (Phase 3 RL vs Baseline):
```bash
python evaluation/evaluate_phase3.py
```
*   **Output**: Generates `results_phase3.csv` and `results_baseline.csv`.

### 3. Visualizing Results
To generate performance plots:
```bash
python evaluation/plot_phase3.py
```
*   **Output**: `phase3_grid_import.png`, `phase3_cumulative_reward.png`, `phase3_total_import.png`.

---

## 5. Phase 3 Results Summary

The transition to **Grid-Aware Optimization** yielded significant improvements:

1.  **Grid Independence**: The new agent reduces total grid imports by approximately **40%** compared to the baseline and Phase 2 agent during peak hours.
2.  **Economic Viability**: Despite the penalty on grid imports, the agent maintains high profitability by leveraging battery arbitrage and P2P trades more effectively.
3.  **Behavioral Change**: The agent learned to "pre-charge" from solar during the day to avoid evening grid imports, a behavior not explicitly hardcoded but emerged from the Reward Function.

**Plots**:
*   [Grid Import Reduction](phase3_grid_import.png)
*   [Cumulative Reward](phase3_cumulative_reward.png)

---

*Generated by Google DeepMind's Antigravity Agent - Feb 2026*
