# P2P-RL-Model: Mathematical Formulation & Weights

This document details the mathematical models, formulas, and specific weight configurations used in the P2P-RL-Model project. It covers the Market Mechanism, Physical Environment logic, Reward Function components, and the Reinforcement Learning hyperparameters.

## 1. Market Clearing (Uniform Price Double Auction)
The `MatchingEngine` (in `market/matching_engine.py`) determines the clearing price and energy allocation.

### Matching Logic
1.  **Sell Orders** are sorted by Price (Ascending).
2.  **Buy Orders** are sorted by Price (Descending).
3.  The **Clearing Price ($P^*$)** is found where Supply $\approx$ Demand.

**Formulas:**
*   **Marginal Match Price**: When a buyer ($B$) and seller ($S$) match:
    $$P^* = \frac{P_{buyer} + P_{seller}}{2}$$
*   **Default Price**: If no trades occur internally:
    $$P^* = \frac{P_{grid\_buy} + P_{grid\_sell}}{2}$$

### Grid Interaction Rules
Unmatched residual energy interacts with the main grid based on price limits:
*   **Export to Grid**: Occurs if $Ask\_Price \leq P_{feed-in}$.
*   **Import from Grid**: Occurs if $Bid\_Price \geq P_{retail}$.

**Weights (Price Constants):**
| Parameter | Variable Name | Value | Description |
| :--- | :--- | :--- | :--- |
| **Grid Retail Price** | `grid_buy_price` | **$0.20 / kWh** | Cost to buy from the grid. |
| **Grid Feed-in Tariff** | `grid_sell_price` | **$0.10 / kWh** | Revenue for selling to the grid. |

---

## 2. Environment Physics
The `EnergyMarketEnvRobust` (in `train/energy_env_robust.py`) enforces physical constraints.

### Battery Dynamics
State of Charge ($SoC$) updates are governed by efficiency ($\eta$) and power limits ($P_{max}$).

**Formulas:**
*   **Charging ($P_{in} > 0$):**
    $$SoC_{t+1} = SoC_t + (P_{in} \cdot \Delta t \cdot \sqrt{\eta})$$
    
*   **Discharging ($P_{out} > 0$):**
    $$SoC_{t+1} = SoC_t - \left( \frac{P_{out} \cdot \Delta t}{\sqrt{\eta}} \right)$$

*   **Effective Action:**
    The request is clipped to physical availability:
    $$P_{charge} = \min(P_{req}, P_{max\_rate}, \frac{Capacity - SoC_t}{\Delta t \cdot \sqrt{\eta}})$$
    $$P_{discharge} = \min(P_{req}, P_{max\_rate}, \frac{SoC_t \cdot \sqrt{\eta}}{\Delta t})$$

**Weights (Physics Parameters):**
| Parameter | Variable Name | Value | Description |
| :--- | :--- | :--- | :--- |
| **Battery Capacity** | `battery_capacity_kwh` | **50.0 kWh** | Maximum storage. |
| **Max Power Rate** | `battery_max_charge_kw` | **25.0 kW** | Max charge/discharge per step. |
| **Round-trip Efficiency** | `battery_roundtrip_eff` | **0.95** (95%) | $\eta$ described above. |
| **Timestep Duration** | `timestep_hours` | **1.0 hour** | Duration of one simulation step. |
| **Grid Line Limit** | `max_line_capacity_kw` | **200.0 kW** | Max aggregate export/import. |

---

## 3. Reward Function (RL Objective)
The reward function (in `train/reward_tracker.py` and `energy_env_robust.py`) is a composite scalar signal designed to balance profit, safety, and fairness.

**Total Reward Formula:**
$$R_{total} = \sum_{i=1}^{N} (Profit_i - Costs_i) - Penalty_{fairness}$$

### Component Computations

1.  **Financial Profit ($Profit_i$):**
    $$Profit_i = (Q_{trade} \cdot P_{market} \cdot \Delta t) - (Q_{curtailed} \cdot P_{market} \cdot \Delta t \cdot 1.5)$$
    *Note: Curtailment is penalized at 1.5x the potential revenue lost.*

2.  **CO2 Penalty ($C_{CO2}$):**
    Applied to imports when grid is "dirty".
    $$C_{CO2} = Q_{import} \cdot I_{CO2} \cdot \lambda_{CO2}$$
    *   $I_{CO2}$: Carbon intensity (mean **0.4**).
    *   $\lambda_{CO2}$ (`co2_penalty_coeff`): **1.0**.

3.  **Grid Overload Penalty ($C_{grid}$):**
    Penalizes usage when total line flow exceeds capacity.
    $$C_{grid} = \frac{|Q_{agent}|}{Q_{total}} \cdot (Q_{total} - Q_{limit}) \cdot \lambda_{overload}$$
    *   $\lambda_{overload}$ (`overload_multiplier`): **50.0** (Strong safety constraint).

4.  **Battery Wear Cost ($C_{batt}$):**
    $$C_{batt} = E_{throughput} \cdot \lambda_{batt}$$
    *   $\lambda_{batt}$ (Fixed constant): **0.02**.

5.  **SoC Smoothing ($C_{soc}$):**
    Encourages maintaining battery reserves around 50%.
    $$C_{soc} = 0.1 \cdot \left(\frac{SoC - 0.5 \cdot Cap}{Cap}\right)^2$$
    *   Weight: **0.1** (Soft constraint).

6.  **Fairness Penalty ($Penalty_{fairness}$):**
    Uses Gini Coefficient ($G$) to penalize profit inequality.
    $$Penalty_{fairness} = \lambda_{fairness} \cdot G(Profits) \cdot (1 + 0.05 \cdot |Q_{total\_export}|)$$
    *   $\lambda_{fairness}$ (`fairness_coeff`): **0.5**.

**Summary of Reward Weights:**
| Component | Weight Variable | Value |
| :--- | :--- | :--- |
| **CO2 Penalty** | `co2_penalty_coeff` | **1.0** |
| **Grid Overload** | `overload_multiplier` | **50.0** |
| **Battery Wear** | (hardcoded) | **0.02** |
| **SoC Smoothing** | (hardcoded) | **0.1** |
| **Curtailment** | (hardcoded multiplier) | **1.5** |
| **Fairness** | `fairness_coeff` | **0.5** |

---

## 4. Reinforcement Learning Hyperparameters
The PPO algorithm is configured in `train/train_sb3_ppo.py` with the following parameters.

### Network Architecture
*   **Policy Network (Action)**: [400, 300] units (MLP).
*   **Value Network (Critic)**: [400, 300] units (MLP).
*   **Input**: Flattened observation vector of size $N\_agents \times (6 + 4 \times H_{forecast})$.

### Training Config
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | **3e-4** | Linearly decaying to 0. |
| **n_steps** | **4096** | Timesteps per environment per update. |
| **batch_size** | **256** | Minibatch size for optimization. |
| **n_epochs** | **10** | Optimization epochs per update. |
| **gamma** ($\gamma$) | **0.99** | Discount factor. |
| **gae_lambda** ($\lambda$) | **0.95** | GAE smoothing factor. |
| **clip_range** | **0.2** | PPO clipping parameter. |
| **ent_coef** | **0.01** | Entropy coefficient (Encourages exploration). |
| **Timesteps** | **100,000** | Default total training steps. |

### Normalization
*   **Observation Clipping**: [-10, 10]
*   **Reward Clipping**: [-10, 10]
*   **Normalization**: Standard score ($z = \frac{x - \mu}{\sigma}$) applied to both obs and rewards.

---

## 5. Operational Thresholds & Limits
These hard constraints are enforced during training (`energy_env_improved.py`) to ensure stability and physical realism.

### Training Safety (Normalization)
*   **Observation Clipping (Limit)**: `[-10.0, +10.0]` (Std Dev units). Prevents extreme inputs.
*   **Reward Clipping (Limit)**: `[-10.0, +10.0]` (Std Dev units). Stabilizes gradient updates.

### Action Space Limits
The agent's output is immediately clipped to these bounds:
*   **Battery Power**: `[-25.0, +25.0] kW` (Negative=Discharge, Positive=Charge)
*   **Grid Trade**: `[-200.0, +200.0] kW` (Negative=Buy, Positive=Sell)
*   **Price Bid**: `[0.0, 1.0]` (Normalized range)

### Physical & Randomization Thresholds
| Parameter | Limit Value | Description |
| :--- | :--- | :--- |
| **Grid Line Capacity** | **200.0 kW** | Max aggregate flow allowed before penalties/safety kick in. |
| **Per-Agent Export** | **30.0 kW** | Soft limit. Exports exceeding this incur a penalty (`export_penalty_coeff`). |
| **Max Demand (Random)** | **500.0 kW** | Maximum cap during random walk generation to prevent explosion. |
| **Max PV (Random)** | **200.0 kW** | Maximum cap during random walk generation. |
| **Max CO2 Intensity** | **2.0 kg/kWh** | Hard cap on carbon intensity values. |
