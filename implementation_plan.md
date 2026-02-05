# Implementation Plan: Environment Upgrade (Phase 1)

## Goal
Upgrade `EnergyMarketEnv` to support full state requirements: CO2 intensity, Forecast Uncertainty, and Line Flows.

## Proposed Changes

### 1. `train/energy_env_improved.py`

#### [MODIFY] `EnergyMarketEnv` Class

*   **`__init__`**:
    *   Add `co2_intensity_mean` (g/kWh) and `co2_intensity_std`.
    *   Add `forecast_uncertainty_std` (noise level for forecasts).
*   **Observation Space**:
    *   Expand to include:
        *   `CO2_grid` (1 float)
        *   `Line_Flows` (1 float per line? Or just max/total? Req says `F_lines`. We have 1 main line constraint in this simplified grid, so maybe just `total_export`, `total_import` is enough, but we can add `line_loading_pct`).
        *   `Forecast_Uncertainty` (1 float per forecast step).
*   **`reset`**:
    *   Initialize CO2 intensity (random walk or fixed profile).
*   **`step`**:
    *   Update CO2 intensity (random walk).
    *   Calculate `co2_penalty = grid_import * co2_intensity * penalty_factor`.
    *   Calculate `renewable_bonus = (self_consumption + export) * bonus_factor`.
    *   Update Reward: `reward -= co2_penalty`, `reward += renewable_bonus`.
    *   Update Observation with new features.

### 2. `train/energy_env_improved.py` (Forecast Logic)

*   **`_naive_forecast`**:
    *   Return tuple `(mean, std)`.
    *   Add noise to the "mean" part to simulate forecast error.
    *   "std" part can be constant or proportional to horizon.

## Verification Plan
*   Run `debug_eval.py` (updated) to check observation shape and reward components.
