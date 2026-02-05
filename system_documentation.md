# P2P-RL Energy Trading System: Technical Documentation

This document details the constraints, working mechanisms, and decision-making logic of the Reinforcement Learning (RL) based Peer-to-Peer (P2P) energy trading system.

## 1. System Constraints

The model operates under strict physical and operational constraints to ensure safety, feasibility, and grid stability. These are enforced by the **Environment** and the **Safety Filter**.

### A. Prosumer (Agent) Constraints
*   **Battery Capacity**: An agent cannot store more energy than `battery_capacity_kwh` (e.g., 50 kWh).
*   **State of Charge (SoC)**:
    *   **Minimum**: SoC cannot drop below 0%.
    *   **Maximum**: SoC cannot exceed 100%.
    *   **Logic**: If an agent tries to discharge an empty battery, the `FeasibilityFilter` overrides the action to 0.
*   **Power Rates**:
    *   **Charge/Discharge Limit**: The battery cannot charge or discharge faster than `battery_max_charge_kw` (e.g., 25 kW) in a single timestep.
*   **Simultaneous Operation**: A battery cannot charge and discharge at the same time. The net action is a single scalar (positive=charge, negative=discharge).

### B. Consumer Constraints
*   **Demand Satisfaction**: Consumer demand is treated as a "hard" requirement. It must be met by:
    1.  Local PV Generation.
    2.  Battery Discharge.
    3.  P2P Imports.
    4.  Grid Imports (Last resort).
*   **Unmet Demand**: If all sources fail (e.g., grid outage simulation), it is recorded as "Loss of Load," but in normal operation, the Grid acts as an infinite slack bus.

### C. Grid & Market Constraints
*   **Line Capacity**: The physical transmission lines have a limit (`max_line_capacity_kw`, e.g., 200 kW).
    *   **Overload**: If total exports or imports exceed this, the system applies a heavy **penalty** to the agent's reward to discourage this behavior.
*   **Power Balance**: At every timestep, `Supply` must equal `Demand`.
    *   `Generation + Imports + Discharge = Consumption + Exports + Charge + Losses`.
    *   **Strict Enforcement**: In `EnergyMarketEnvRobust`, this is verified at runtime with assertions (`1e-5` tolerance).

### D. Advanced Modularization (Phase 9)
*   **EnergyMarketEnvRobust**: A refactored, research-grade implementation in `train/energy_env_robust.py`.
    *   **Modular Steps**: Physics, Market, and Reward logic are decoupled.
    *   **RewardTracker**: Detailed component-wise logging (Profit vs Penalty).
    *   **Security**: Integrated `FeasibilityFilter` and `Conservation Checks`.

---

## 2. Working Mechanism

The system is composed of three main layers: The **RL Agent**, the **Safety Layer**, and the **Market Engine**.

### Step 1: Observation (The "Eyes")
The RL Agent observes the current state of the world:
*   **Internal State**: Current Demand, Battery SoC, Solar PV generation.
*   **Market State**: Current Grid Prices (Retail/Feed-in), CO2 Intensity.
*   **Forecasts**: Predictions for future Demand and PV (with uncertainty).

### Step 2: Decision (The "Brain")
Based on the observation, the Agent outputs an **Action** vector with three components:
1.  **Battery Action (kW)**: "Charge 5kW" or "Discharge 10kW".
2.  **Trade Quantity (kW)**: "Sell 15kW" or "Buy 8kW".
3.  **Price Bid ($/kWh)**: "I am willing to trade at $0.15/kWh".

### Step 3: Safety Check (The "Guardrails")
The **FeasibilityFilter** intercepts the action before it hits the market:
*   *Agent says*: "Discharge 50kW" (but battery only has 10kWh and max rate is 25kW).
*   *Filter corrects*: "Discharge 10kW" (Max available).
*   *Result*: The safe, physically possible action is executed.

### Step 4: Market Matching (The "Handshake")
The **MatchingEngine** collects all safe bids/asks:
*   **Sellers** are sorted by their Ask Price (Low to High).
*   **Buyers** are sorted by their Bid Price (High to Low).
*   **Matching**: A trade occurs if a Buyer is willing to pay *at least* what a Seller is asking.
*   **Clearing Price**: Determined by the intersection of Supply and Demand.

### Step 5: Grid Settlement (The "Backup")
*   **Surplus**: If an agent wants to sell but finds no P2P buyer, they sell to the Grid at the (lower) Feed-in Tariff.
*   **Deficit**: If an agent wants to buy but finds no P2P seller, they buy from the Grid at the (higher) Retail Rate.

---

## 3. Energy Allocation Logic

The allocation follows a strict hierarchy of value:

1.  **Self-Consumption**:
    *   *Why?* It's free. Using your own PV to meet your own Demand is always the most efficient (zero transmission loss, zero cost).
2.  **P2P Trading**:
    *   *Why?* It's cheaper than the Grid.
    *   Buyers pay less than Retail Rate.
    *   Sellers earn more than Feed-in Tariff.
    *   The RL agent learns to set prices *between* these two bounds to maximize probability of a trade.
3.  **Battery Storage**:
    *   *Why?* Time-shifting.
    *   If prices are low (sunny afternoon), charge the battery.
    *   If prices are high (evening peak), discharge to sell or consume.
4.  **Grid Interaction**:
    *   *Why?* Reliability.
    *   Used only when local and P2P resources are exhausted (Import) or when batteries are full (Export).

---

## 4. Why does the model choose this path?

The RL Agent is trained to maximize a **Reward Function**. Its decisions are driven by the incentives we programmed:

### A. Profit Maximization (Economic Driver)
*   **Behavior**: The agent learns to "Buy Low, Sell High."
*   **Example**: It charges the battery during the day (when PV is free or prices are low) and discharges in the evening (when it can sell to neighbors at a high price).
*   **Price Bidding**: It learns that bidding too high means no one buys (zero profit), and bidding too low means leaving money on the table. It converges to a competitive market price.

### B. CO2 Reduction (Environmental Driver)
*   **Incentive**: We added a `co2_penalty` for importing from the grid when carbon intensity is high.
*   **Behavior**: If the grid is "dirty" (high CO2), the agent will prefer to discharge its battery or buy from a neighbor with solar, even if it costs slightly more, to avoid the penalty.

### C. Grid Stability (Safety Driver)
*   **Incentive**: We added an `overload_penalty` for exceeding line capacity.
*   **Behavior**: If many agents try to export at once, the lines overload. The agent learns to coordinate (implicitly) or store energy locally to avoid the penalty.

### Summary
The model chooses its path because it has learned, through thousands of trial-and-error episodes, that balancing **Self-Sufficiency**, **Strategic Trading**, and **Battery Arbitrage** yields the highest long-term reward.

### Safety vs Learning (Examiner Note)
- **Enforced by deterministic layers**: SoC bounds, battery power limits, trade feasibility (surplus/deficit), and OOD fallback. These are **not** learned; they are guaranteed by the FeasibilityFilter and SafetySupervisor.
- **Learned by the RL agent**: When to charge/discharge and how much to bid (quantity and price). The agent can output infeasible actions; the safety layer corrects them before execution.

---

## 5. Autonomous Architecture (Phase 10)

The system has been upgraded to a **3-Layer Autonomous Stack** to ensure safety and regulatory compliance without human intervention.

1.  **Layer 1 (RL)**: Proposes strategic intent.
2.  **Layer 2 (Optimization)**: Deterministically clips actions to physical limits.
3.  **Layer 3 (Safety Supervisor)**: Enforces hard invariants (SoC bounds, Conservation) and triggers **Fallback (Grid-Only)** if anomalies (OOD) are detected.

For details, see `autonomous_architecture.md`.
