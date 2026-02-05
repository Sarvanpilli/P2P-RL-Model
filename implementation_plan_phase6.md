# Implementation Plan: Advanced Pricing & Market Logic (Phase 6)

## Goal
Enable the RL agent to set **Dynamic Price Bids** (Limit Orders) in addition to quantities, creating a fully autonomous trading agent. Blockchain integration is deferred.
    
## Proposed Changes

### 1. [MODIFY] `market/matching_engine.py`

*   **`match(orders)`**: Update signature to accept `bids` (list of `[qty, price]`).
*   **Logic Upgrade**:
    *   **Limit Order Book**:
        *   Sellers: Willing to sell `qty` at `price` or higher.
        *   Buyers: Willing to buy `qty` at `price` or lower.
    *   **Matching**:
        *   Sort Sellers (ASC price) and Buyers (DESC price).
        *   Match where `Buyer_Price >= Seller_Price`.
        *   **Clearing Price**: Can be the mid-point or the seller's price (Pay-as-Bid vs Uniform). We will use **Uniform Pricing** (intersection point) for stability.
    *   **Grid Interaction**:
        *   Unmatched Surplus -> Sold to Grid at `feed_in_tariff` (if `price_bid <= feed_in`).
        *   Unmatched Deficit -> Bought from Grid at `retail_rate` (if `price_bid >= retail`).

### 2. [MODIFY] `train/energy_env_improved.py`

*   **Action Space**: Expand to `3 * n_agents`.
    *   Index 0: `battery_kw`
    *   Index 1: `trade_qty_kw` (Positive=Sell, Negative=Buy)
    *   Index 2: `price_bid_$/kWh` (0.0 to 1.0)
*   **Step Logic**:
    *   Extract `price_bid` from action.
    *   Pass `(trade_qty, price_bid)` to `self.matching_engine.match()`.

## Verification Plan
*   **Unit Tests**: Update `tests/test_market.py` to verify Limit Order logic (e.g., a high sell bid should NOT clear if demand is low price).
*   **Training**: Retrain the agent to learn pricing strategies.
