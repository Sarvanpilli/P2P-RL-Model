# Implementation Plan: Advanced Matching Engine (Phase 3)

## Goal
Upgrade `MatchingEngine` to support **Double Auction** logic and integrate it into `EnergyMarketEnv`, replacing the simplified pro-rata logic.

## Proposed Changes

### 1. [MODIFY] `market/matching_engine.py`

*   **`match(orders)`**:
    *   Input: `orders` array (positive=sell, negative=buy).
    *   Logic:
        *   Separate Buys and Sells.
        *   **Clearing Price**: Find the intersection of Supply and Demand curves.
            *   Since we don't have explicit price bids yet (agents are price takers or use fixed bids), we can simulate a "Uniform Price Auction" where the price is determined by the net imbalance (similar to current logic but formalized).
            *   OR, if we want to support future price bids, we structure it to accept `(quantity, price)` tuples.
        *   **Matching**:
            *   If Supply > Demand: All buyers matched. Sellers pro-rata curtailed. Price drops.
            *   If Demand > Supply: All sellers matched. Buyers pro-rata curtailed. Price rises.
        *   **Grid Interaction**: The Grid acts as the "Market Maker of Last Resort".
            *   Net Surplus -> Sold to Grid (at `price_sell_to_grid`).
            *   Net Deficit -> Bought from Grid (at `price_buy_from_grid`).
    *   Output: `trades` (matched quantities), `clearing_price`, `grid_import`, `grid_export`.

### 2. [MODIFY] `train/energy_env_improved.py`

*   Import `MatchingEngine`.
*   Initialize `self.matching_engine`.
*   In `step()`:
    *   Replace the manual `intended_injection_kw` processing code with `self.matching_engine.match()`.
    *   Use the returned `trades`, `price`, and `grid_flow` to calculate rewards and state updates.

## Verification Plan
*   Create `tests/test_market.py` to verify auction logic (e.g., surplus leads to export, deficit leads to import).
*   Run `debug_eval.py` to ensure integration works.
