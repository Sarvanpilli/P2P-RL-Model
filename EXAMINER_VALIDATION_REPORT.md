# Academic Validation Report: Autonomous P2P Energy Trading using MARL

**Project**: Final Year Project (16 credits)  
**Title**: Autonomous Peer-to-Peer Energy Trading using Multi-Agent Reinforcement Learning  
**Purpose**: Critical validation, corrections, and examiner-safe documentation for report and viva defence.

---

## 1. Problem–Solution Alignment

### Stated Problem (Expected)
- Autonomous P2P energy trading in a microgrid.
- Use of real smart-meter data (e.g. Ausgrid Solar Home Dataset).
- Battery storage, grid constraints, P2P market, safety layer.
- Time-of-Use pricing, liberalised bidding, battery degradation, ramp-rate, quadratic distribution losses.

### Implemented vs Claimed

| Requirement | Implemented | Notes |
|-------------|-------------|--------|
| Custom Gymnasium env for P2P | Yes | `EnergyMarketEnvRobust` in `train/energy_env_robust.py` |
| Multi-agent RL (PPO) | Yes | Single shared PPO policy, flattened obs/action |
| Real data replay | Conditional | **Evaluation** can use `evaluation/ausgrid_p2p_energy_dataset.csv`. **Training** default is `test_day_profile.csv` (synthetic). See §4. |
| Battery + grid constraints | Yes | `MicrogridNode`, line capacity, overload penalty |
| P2P market | Yes | `MatchingEngine`: uniform-price double auction, limit orders |
| Deterministic safety layer | Yes | `AutonomousGuard` → `FeasibilityFilter` + `SafetySupervisor` |
| Time-of-Use pricing | Yes | `_get_grid_prices()`: Off-Peak / Standard / Peak |
| Liberalised bidding | Yes | Price bid in action; matching by price priority |
| Battery degradation cost | Yes | `RewardTracker`: throughput × unit cost |
| Ramp-rate constraints | Code only | `MicrogridNode.set_ramp_limit()` and `_apply_ramp_constraint()` exist but **are not invoked** by `EnergyMarketEnvRobust`. Optional; do not claim “enforced” unless enabled. |
| Quadratic distribution losses | No (Robust env) | Present in `energy_env_improved` only. Do not claim for the main (Robust) pipeline. |

**Recommendation**: In the report, state exactly which env and data are used for each experiment (e.g. “evaluation used EnergyMarketEnvRobust with Ausgrid-derived CSV and 1 h timesteps”). For ramp-rate and distribution losses, either implement in Robust and document, or remove from claimed features and list as “future work”.

---

## 2. Technical Correctness

### 2.1 Units (Corrected in Code)

- **Power**: kW (instantaneous). Used for battery power, trade quantity, line flows.
- **Energy**: kWh. SoC and throughput are in kWh. Per-step energy = power × `timestep_hours`.
- **Carbon**: Environment uses `co2` in kg/kWh. `step_carbon_mass` (kg) = `import_kwh × co2`, with `import_kwh = total_import * timestep_hours`. **Bug fix applied**: previously `total_import` (kW) was multiplied by `co2` without `timestep_hours`.
- **Price**: $/kWh. Profit per step = trade (kW) × price × `timestep_hours` → $.

### 2.2 Time Resolution

- **Timestep**: 1 hour (`timestep_hours = 1.0`) in training and evaluation unless changed.
- **Episode length**: Determined by data (e.g. 24 h or full CSV length). Evaluation uses `random_start_day=False` for reproducible replay.

### 2.3 Energy Balance

- **MicrogridNode**: Net load (kW) = demand − PV + charge − discharge. Positive ⇒ import need.
- **MatchingEngine**: Sum of trades equals grid flow; conservation check in code.
- **Environment**: Power balance is implicit via feasible actions and market clearing; no explicit global equation in Robust env. Acceptable for report if stated as “implicit through feasibility filter and market clearing”.

### 2.4 Corrections Applied in Codebase

1. **`_get_current_data()` fallback**: Return order was `(pv, dem, co2)` but callers expect `(demand, pv, co2)`. Corrected to `(dem, pv, co2)`.
2. **Carbon mass**: `step_carbon_mass = total_import * co2` → `import_kwh = total_import * timestep_hours`, `step_carbon_mass = import_kwh * co2`.
3. **Evaluation CSV**: Added `grid_flow`, `total_carbon_kg`, `cumulative_carbon_kg`, and per-agent `agent_{i}_soc` so plots and reports have correct units and series.
4. **`eval_per_agent.py`**: Switched to `EnergyMarketEnvRobust`, configurable model/data paths, and info keys (`trades_kw`, `battery_throughput_delta_kwh`, `line_overload_kw`).
5. **`MicrogridNode`**: Removed duplicate `step()` definition; kept single implementation with optional ramp; initialised `last_power_kw` and `max_ramp_kw` so ramp logic does not error when ramp is disabled.

---

## 3. Safety & Learning Separation

### What Is Enforced (Deterministic, Not Learned)

- **FeasibilityFilter**: Clips battery power to SoC and rate limits; clips trade to physical surplus/deficit. Idempotent.
- **SafetySupervisor**: OOD detection on observations; fallback to safe action (e.g. no trade) when triggered.
- **MicrogridNode**: SoC hard clip to [0, capacity]; charge/discharge limited by capacity and rate.
- **Market**: Clearing and grid settlement are rule-based.

**Report wording**: “Feasibility and safety (SoC bounds, power limits, trade feasibility) are **enforced by deterministic layers**. The RL agent **learns** when to charge/discharge and how to bid; it does **not** learn to satisfy physics or safety.”

### What Is Learned (RL)

- Battery charge/discharge timing and magnitude (within feasible set).
- P2P quantity and price bid to maximise reward (profit, CO2, congestion penalties).
- No learning of constraint satisfaction; the agent can output infeasible actions that are then corrected.

**Avoid**: “The agent learns to stay within SoC limits” or “RL ensures feasibility.” **Use**: “The agent learns a trading strategy; feasibility is guaranteed by the safety layer.”

---

## 4. Dataset Usage (Examiner-Safe Wording)

### Data Sources in the Repo

1. **`test_day_profile.csv`** (root): **Synthetic**. Generated by `generate_data.py` (Gaussian-style PV/demand). Used by default in training.
2. **`evaluation/ausgrid_p2p_energy_dataset.csv`**: **Derived from or aligned with real-world smart-meter style data** (e.g. Ausgrid Solar Home Dataset). Column format: `hour`, `agent_{i}_pv_kw`, `agent_{i}_demand_kw`. Used in evaluation when path is provided (e.g. `evaluation/evaluate_episode.py --data evaluation/ausgrid_p2p_energy_dataset.csv`).

### Replay Mechanism

- **Time-series replay**: The environment indexes the CSV by `current_idx` and reads the row for that timestep. No sampling of rows at random; order is sequential. For evaluation, `random_start_day=False` gives deterministic replay from the start of the file.
- **Units in CSV**: Power in kW (per hour). SoC and energy are computed in the env using `timestep_hours`.

### Examiner-Safe Statement (for Report)

- “Demand and PV profiles are supplied as **time-series data** (hourly resolution). For **evaluation**, we use a dataset derived from real smart-meter data (Ausgrid Solar Home Dataset style) and replay it **sequentially** without shuffling. For **training**, the default profile is synthetic; experiments can be repeated with the same evaluation dataset for comparability.”
- Do **not** claim “all experiments use raw Ausgrid data” unless every run explicitly uses the Ausgrid-derived CSV.

---

## 5. Evaluation Completeness

### Present

- **Baseline**: “Grid-only, no battery” in `evaluation/generate_plots.py` and `generate_professional_report.py`: net = demand − PV; buy/sell at ToU retail/feed-in. Compare cumulative cost/profit with P2P-RL.
- **System-level metrics**: Total import/export (kWh), cumulative carbon (kg), market clearing price, reward components (profit, CO2, grid penalty, battery cost).
- **Market**: Clearing price and volume from `MatchingEngine`; logged in evaluation CSV.

### Optional Additions (Minimal)

- **Explicit baseline script**: One script that (1) loads the same CSV, (2) runs the baseline rule (no battery, no P2P), (3) writes a baseline CSV (e.g. hourly cost) so “Baseline vs P2P-RL” plots use identical data and units.
- **One sentence in report**: “Baseline is grid-only with the same demand/PV time series and ToU tariffs.”

---

## 6. Visual Evidence (Plots)

### Required Plots and Checks

| Plot | Purpose | Units / Checks |
|------|--------|----------------|
| **Battery SoC vs Time** | SoC stays within bounds and shows arbitrage behaviour | Y-axis: **kWh** (or % if scale 0–100 and caption states “SoC (%)”). X-axis: hour or step. |
| **Grid Import/Export vs Time** | Community grid interaction | Y-axis: **Power (kW)** for instantaneous, or **Energy (kWh)** if summed over step; state which. X-axis: hour. |
| **Market Clearing Price vs Hour** | P2P price vs grid tariffs | Y-axis: **$/kWh**. X-axis: hour of day. Compare to ToU retail/feed-in. |
| **Cumulative Profit (Baseline vs P2P-RL)** | Economic value of P2P-RL | Y-axis: **$**. Same time base and data for both curves. Baseline = no battery, no P2P. |

### Validation

- **plot_results.py**: Uses `step`, `market_price`, `grid_flow` (or total_export − total_import), `agent_*_soc`, `cumulative_carbon_kg` / `total_carbon_kg`. After code fixes, evaluation CSV contains these.
- **generate_plots.py**: Uses evaluation_results.csv and data CSV for baseline; paths made relative. Profit comparison uses same length and hourly ToU.

**Claim**: “Plots use consistent units (kW/kWh, $, kg CO2) and the same time series for baseline and P2P-RL.”

---

## 7. Claim Rewriting (Conservative & Precise)

### Before (Risky)

- “The system uses real Ausgrid data.”
- “The RL agent learns to respect battery limits.”
- “We implement ramp-rate constraints and distribution losses.”
- “P2P-RL always improves profit over baseline.”

### After (Examiner-Safe)

- “**Evaluation** uses a time-series dataset derived from real smart-meter data (Ausgrid-style). **Training** default is a synthetic profile; both are replayed sequentially.”
- “Battery and trade **feasibility** are enforced by a **deterministic** feasibility filter and safety supervisor. The **RL agent** learns **trading strategy** (when to charge/discharge and how to bid).”
- “The **main environment** (EnergyMarketEnvRobust) uses ToU pricing, limit-order P2P matching, battery degradation cost in the reward, and optional ramp-rate logic in the physics module; **ramp-rate is not currently enabled** in training. **Quadratic distribution losses** are implemented in an alternative env only; not in the Robust pipeline.”
- “Under the evaluated scenarios and baseline (grid-only, same demand/PV), the **P2P-RL system** achieves [X]% improvement in cumulative community profit (or equivalent metric). Results depend on data, hyperparameters, and seed; no claim of universal superiority.”

### Explicit Limitations (for Report / Viva)

1. **Single policy**: One shared PPO policy for all agents; not independent learners.
2. **Data**: Evaluation can use Ausgrid-derived data; training default is synthetic.
3. **Ramp-rate**: Implemented in `MicrogridNode` but not enabled in the main env.
4. **Distribution losses**: Not in EnergyMarketEnvRobust.
5. **Baseline**: Grid-only, no battery; no other baselines (e.g. rule-based battery) unless implemented.
6. **Scale**: Evaluated with small N (e.g. 4 agents); scaling to large N may require further tuning.
7. **Safety**: Guaranteed by deterministic layers, not by the learned policy.

---

## 8. Summary of Code and Doc Changes

- **train/energy_env_robust.py**: Fallback data order fixed; carbon mass uses `timestep_hours`; info extended with `trades_kw`, `battery_throughput_delta_kwh`, `line_overload_kw`.
- **evaluation/evaluate_episode.py**: CSV now includes `grid_flow`, `total_carbon_kg`, `cumulative_carbon_kg`, `agent_{i}_soc`; CLI for model/data/output; paths relative.
- **evaluation/plot_results.py**: `grid_flow` derived from total_export/import if missing; CO2 from `cumulative_carbon_kg` or `total_carbon_kg`.
- **evaluation/generate_plots.py**, **evaluation/generate_professional_report.py**: Paths relative to script directory.
- **train/eval_per_agent.py**: Uses EnergyMarketEnvRobust, CLI, VecNormalize, and info keys from Robust; curtail_events removed (not in Robust).
- **simulation/microgrid.py**: Duplicate `step()` removed; ramp attributes initialised so optional ramp does not error.

Use this report to align the written report and viva with the implementation and to avoid overclaiming.
