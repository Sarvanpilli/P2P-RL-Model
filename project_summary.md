# P2P-RL-Model: Project Overview & Implementation Guide

## 1. Project Description

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system for **Peer-to-Peer (P2P) Energy Trading** in a simulated residential microgrid. The goal is to create an autonomous framework of intelligent agents that manage distributed energy resources — Battery Storage and Solar PV — to maximize economic profit, minimize CO₂ emissions, and ensure grid stability through decentralized, market-based coordination.

The system simulates a microgrid with **4 heterogeneous prosumer households** (Solar PV, Wind, Electric Vehicle/V2G, and Standard Load) each equipped with a battery storage unit. Agents can trade energy with each other (P2P) or with the main grid, using a dynamic pricing mechanism — a **Uniform Price Double Auction** — learned by the reinforcement learning policy.

This integrated system — combining the MARL agents, dynamic market mechanism, curriculum-based training, and a two-tier safety architecture — is referred to as the **SLIM (Safety-constrained Liquidity-Integrated Market)** framework.

---

## 2. Key Features & Innovations

### A. Advanced Market Logic — Uniform Price Double Auction
- **Old baseline**: Simple quantity-based P2P matching with fixed prices.
- **SLIM Market**: Agents submit **Limit Orders** (Quantity + Price Bid). The matching engine sorts buyers (high-to-low) and sellers (low-to-high) to find a single **Clearing Price** where supply meets demand. This replicates real-world electricity market microstructure.
- Grid interaction is conditional: unmatched surplus exports to grid only if `ask_price ≤ feed-in tariff`; agents import only if `bid_price ≥ retail tariff`.

### B. Curriculum-Based Training to Break Nash Equilibrium Collapse
- **Problem identified**: Initial SLIM training produced a Nash equilibrium collapse — all four agents converged to homogeneous all-seller strategies, yielding zero P2P volume.
- **Root cause**: Symmetric reward structure where selling to the grid was individually rational regardless of others' actions.
- **Fix — 3-Stage Curriculum**:
  - **Stage 1** (0–50k steps): Aggressive P2P incentives (`p2p_bonus=0.30`, `no_buyer_penalty=0.20`), elevated entropy (`ent_coef=0.02`) to force exploration across buyer/seller roles.
  - **Stage 2** (50k–150k): Balanced P2P rewards with grid penalties. Lagrangian safety constraints introduced (`α=0.001`).
  - **Stage 3** (150k–300k): Full constraint enforcement (`α=0.005`). Training stabilized by 250k steps.
- **Result**: The market participation metric `market/n_buyers_mean` crossed the 0.5 threshold within the first 50k steps of Stage 1 and remained above it for all subsequent training.

### C. Two-Tier Safety Architecture

> [!NOTE]
> Safety and physics compliance are **guaranteed by deterministic layers** — not learned by the RL agent. The agent learns *trading strategy* (when to charge/discharge and how to bid); the safety layer ensures *all actions are physically feasible*.

**Tier 1 — Projection-Based Hard Constraints (`AutonomousGuard`):**
A deterministic, stateless three-layer projection guarantees zero constraint violations at every timestep:
1. **Jitter / Slew-Rate Clipping**: Bounds rate-of-change of each action dimension per hardware spec (e.g., Solar/Wind battery: 2.5 kW/step; EV: 7.0 kW/step).
2. **Feasibility Filter**: Clips battery actions to exact SoC limits and power ratings; clips trade quantities to physical surplus/deficit.
3. **Hard Veto Fallback**: Out-of-distribution observation detection via `SafetySupervisor`; defaults to a safe no-trade action if triggered.

**Tier 2 — Lagrangian Soft Constraints:**
Three Lagrange multipliers (λ_SoC, λ_line, λ_voltage) are updated via gradient ascent on constraint violations:

```
λ_k ← clip(λ_k + α·(violation_k − threshold_k), 0, λ_max)
```

As multipliers converge, the policy learns to proactively avoid constraint boundaries, reducing hard-projection interventions — confirmed by decreasing `guard_info['layer2_interventions']` counts across successive TensorBoard checkpoints.

### D. GNN Policy Architecture (Research Extension)
- `research_q1/novelty/gnn_policy.py` implements a **GATv2Conv** (Graph Attention Network v2) policy backbone for edge-level interpretability.
- The benchmark models for the 2017 Ausgrid evaluation use a high-performance **MLP policy** (`[400, 300]` units) for maximum training stability. The GNN is available as a research extension for future multi-hop agent coordination experiments.

### E. Environment Enhancements
- **CO₂ Tracking**: Dynamic grid CO₂ intensity. Agents are penalized proportionally for importing dirty grid energy.
- **Forecast Uncertainty**: Observations include noisy short-horizon forecasts for PV and demand (`forecast_horizon=2`), forcing robustness to prediction error.
- **Explicit Physics**: All energy calculations use `dt_hours = 1.0 h` to maintain dimensional consistency: Energy (kWh) = Power (kW) × Time (h).
- **Role Diversity Observation**: A market-balance feature (obs dimension 104 → 105) was added to enable role awareness, supporting the curriculum fix.

---

## 3. Implementation Phases

The project was built through 8 progressive phases:

| Phase | Description |
| :---: | :--- |
| 1 | **Environment Setup** — `EnergyMarketEnv` with battery & grid physics |
| 2 | **Baseline RL** — PPO via Stable-Baselines3 on synthetic data |
| 3 | **Real-Time Wrapper** — Dashboard for visualizing daily agent decisions |
| 4 | **Action Space Expansion** — Added price bidding to agent actions |
| 5 | **Market Engine Upgrade** — Limit Order Book double auction mechanism |
| 6 | **Safety Integration** — `FeasibilityFilter` + `AutonomousGuard` |
| 7 | **Robustness Overhaul** — Unit tests, idempotence checks, conservation tests |
| 8 | **SLIM v2 Final** — Curriculum training, Lagrangian safety, Nash-equilibrium fix |

---

## 4. How to Run the Project

### A. Train the SLIM v2 Agent
```bash
python train/train_sb3_ppo.py --timesteps 300000 --seed 42
```
> Models are saved to `models_slim/`. Training uses `fixed_training_data.csv` (synthetic) by default.

### B. Evaluate on Ausgrid Data (5-Seed Benchmark)
```bash
python research_q1/novelty/run_all_experiments.py
```
> Runs seeds 0–4 on the 1,752-hour evaluation split of the 2017 Ausgrid dataset. Results written to CSV and auto-plotted.

### C. Generate Science Plots
```bash
python research_q1/novelty/plot_results.py
```

### D. Run Robustness Tests
```bash
python tests/test_robustness.py
python -m pytest tests/
```

### E. Monitor Training with TensorBoard
```bash
tensorboard --logdir=tboard_slim/
```
Navigate to `http://localhost:6006` to view reward curves, policy/value loss, and `market/n_buyers_mean`.

### F. Real-Time Dashboard
```bash
python final_viva_dashboard.py
```

---

## 5. Key Results (SLIM v2 — Final Benchmark)

### A. Comparative Performance
All results are averaged over **5 random start-day seeds** on the **1,752-hour evaluation split** (20% holdout, 2017 Ausgrid data).

| Metric | Baseline (No P2P) | Legacy Auction | **SLIM v2 (Proposed)** |
| :--- | :---: | :---: | :---: |
| **P2P Volume (kWh)** | 0.00 ± 0.00 | 67.83 ± 27.29 | **992.77 ± 60.47** |
| **Mean Reward** | — | — | **133.89 ± 5.92** |
| **Buyers per Step** | 0.00 | — | **1.777 ± 0.049** |
| **Safety Violations** | — | — | **0** |

> **P2P Volume Improvement**: **1,363% increase** over the Legacy Auction (992.77 vs 67.83 kWh).  
> **Safety**: Zero constraint violations across all 840 evaluation timesteps.

### B. Scalability (Network Effect)
P2P volume per agent increases monotonically with community size, demonstrating a positive liquidity network effect:

| Agents (N) | Profit / Agent ($) | P2P Volume / Agent (kWh) | Δ from N=4 |
| :---: | :---: | :---: | :---: |
| 4 | −38.05 | 63.66 | — |
| 6 | −36.42 | 71.12 | +11.7% |
| 8 | −35.11 | 82.45 | +29.5% |
| 10 | −33.88 | 94.20 | +48.0% |

> Profit figures are net of CO₂, battery wear, and congestion fees. Gross monetary P2P revenue is positive; net figures reflect deliberate inclusion of social and environmental externality costs.

### C. Verified Dataset

| Property | Value |
| :--- | :--- |
| **Source** | Ausgrid Solar Home Dataset (NSW, Australia) + Regional Wind Data |
| **Year** | 2017 |
| **Timesteps** | 8,760 (hourly, 1 full year) |
| **Train / Eval Split** | 80% Training (7,008 h) / 20% Evaluation (1,752 h) |
| **Normalization** | Min-Max per column, scaled to [0, 1] |
| **Preprocessing Script** | `scripts/preprocess_hybrid_data.py` |

> **Note**: Training default uses synthetic data (`fixed_training_data.csv`). For examiner reproducibility, evaluation always uses the sequential Ausgrid-derived time series with `random_start_day=False`.

---

## 6. Known Limitations

| # | Limitation | Status |
| :---: | :--- | :--- |
| 1 | **Single Shared Policy** — one PPO policy for all agents | By design (reduces computation) |
| 2 | **Ramp-Rate Constraints** — implemented in `MicrogridNode` but not enabled in main env | Future Work |
| 3 | **Quadratic Distribution Losses** — only in `energy_env_improved`, not in the Robust pipeline | Future Work |
| 4 | **Scale** — evaluated with N≤10 agents; large-N may require additional hyperparameter tuning | Future Work |
| 5 | **Training Data** — default training uses synthetic profiles | Evaluation uses real Ausgrid data |

---

## 7. Component Status

| Component | Status | Notes |
| :--- | :---: | :--- |
| **Physics Engine** | ✅ Robust | Energy conserved; `dt_hours` applied consistently. |
| **Market Mechanism** | ✅ Advanced | Uniform Price Double Auction with Limit Orders. |
| **Safety (Tier 1)** | ✅ Verified | Projection filter — hard guarantee of zero violations. |
| **Safety (Tier 2)** | ✅ Active | Lagrangian soft constraints — proactive policy learning. |
| **AI Agent** | ✅ Trained | SLIM v2: 300k curriculum steps; convergence by 250k. |
| **Evaluation** | ✅ Benchmarked | 5-seed, 1,752-hour Ausgrid holdout. |
| **Code Quality** | ✅ High | Modular, unit-tested, reproducible. |
