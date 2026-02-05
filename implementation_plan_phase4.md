# Implementation Plan: Metrics & Reporting (Phase 4)

## Goal
Implement a comprehensive evaluation suite to generate detailed CSV reports and visualization plots for a "Representative Day".

## Proposed Changes

### 1. [NEW] `evaluation/evaluate_episode.py`

Create a script to run a full evaluation episode using the `RealTimeEnvWrapper` (or standard env with fixed seed) and the trained model.

**Features:**
*   **Data Collection**: Record per-step data:
    *   Time
    *   Market Price
    *   Grid Import/Export
    *   Net Imbalance
    *   Per-Agent: Demand, PV, SoC, Battery Action, Grid Trade, Profit, CO2.
*   **CSV Output**: Save `evaluation_results.csv`.
*   **Metrics Calculation**:
    *   Total Profit.
    *   Total CO2 Emissions.
    *   Average Gini Coefficient.
    *   Loss of Load / Curtailment totals.

### 2. [NEW] `evaluation/plot_results.py`

Create a script to visualize the results from `evaluation_results.csv`.

**Plots:**
1.  **Market Dynamics**: Price vs Net Imbalance over 24h.
2.  **Grid Interaction**: Total Import/Export vs Time.
3.  **Agent Behavior**: SoC profiles for all agents.
4.  **CO2 Impact**: Cumulative CO2 emissions.

## Verification Plan
*   Run the evaluation script.
*   Check if CSV and PNG files are generated.
