# Implementation Plan: Robustness & Validation Overhaul (Phase 8)

## Goal
Address the critical audit feedback to make the system robust, safe, and research-ready.

## Proposed Changes

### 1. [MODIFY] `train/safety_filter.py`
*   **Idempotence**: Ensure `filter_action(filter_action(a)) == filter_action(a)`.
*   **Statelessness**: Remove dependency on internal state if possible (pass everything in).
*   **Logic**:
    *   Clip `battery_kw` based on exact `dt_hours` and `soc`.
    *   Clip `trade_kw` based on surplus/deficit after self-consumption.
    *   Clip `price_bid` to `[feed_in, retail]`.
*   **Return**: `safe_action, changed_flag`.

### 2. [MODIFY] `market/matching_engine.py`
*   **Algorithm**: Implement **Uniform Price Double Auction**.
    *   Sort Buyers (DESC) and Sellers (ASC).
    *   Find intersection of cumulative curves.
    *   Clearing Price = Intersection Price.
*   **Conservation**: Add assertion `abs(sum_sold - sum_bought) < 1e-6`.

### 3. [MODIFY] `train/energy_env_improved.py`
*   **Physics**:
    *   Ensure `energy_kwh = power_kw * dt_hours`.
    *   Apply efficiency correctly: `charge * eff` vs `discharge / eff`.
*   **Integration**:
    *   Use `feasibility_filter` output for physics.
    *   Log `filter_changed` count.
*   **Reward**:
    *   Normalize components to `[-1, 1]` range where possible.
    *   Clip final reward.

### 4. [NEW] `tests/test_robustness.py`
*   **Test A**: Energy Balance (Physics).
*   **Test B**: Filter Idempotence.
*   **Test C**: Matching Conservation.
*   **Test D**: Overfit Test (Deterministic Scenario).

## Verification Plan
1.  Run `tests/test_robustness.py` to pass all unit tests.
2.  Run `train_sb3_ppo.py` with `--overfit` flag (to be added) to prove learning capability.
