"""
Phase D: MPC "Gold Standard" Baseline
======================================
Solves the Optimal Power Flow (OPF) problem for a 24-hour horizon using
cvxpy, establishing the Theoretical Maximum Profit / Minimum Cost for the
P2P microgrid. This is the "oracle" upper-bound for RL comparison.

Key design decisions (per Senior Researcher review):
  1. Un-normalization: CSV data is in [0, 1] and is multiplied by
     normalization_config.json max_vals before being fed to the solver.
  2. Battery churn prevention: small throughput cost added to objective.
  3. Terminal SoC constraint: soc_end >= soc_start so each day is fair.
  4. Agent 0 is a pure consumer (no generation data in CSV).
  5. P2P market balance: sum(p2p_sell - p2p_buy) == 0 per timestep.
  6. Physical line capacity enforced.

Usage (from project root):
    python research_q1/baselines/baseline_mpc.py
    python research_q1/baselines/baseline_mpc.py --max-days 7
    python research_q1/baselines/baseline_mpc.py --data-path processed_hybrid_data.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ── resolve project root so imports and relative paths work ──────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    import cvxpy as cp
except ImportError:
    print("ERROR: cvxpy not installed. Run:  pip install cvxpy")
    sys.exit(1)

# ─────────────────────────────── Constants ──────────────────────────────────
N_AGENTS   = 4
HORIZON    = 24          # hours
EFFICIENCY = 0.9         # round-trip sqrt per direction (0.9^0.5 ≈ 0.949)
EFF_CHARGE = EFFICIENCY ** 0.5
EFF_DISCH  = EFFICIENCY ** 0.5

# Battery specs mirroring EnergyMarketEnvRobust._setup_agents
BATTERY_CAP_KWH  = [5.0,  5.0,  62.0,  10.0]   # per agent
MAX_POWER_KW     = [2.5,  2.5,   7.0,   5.0]    # charge/discharge limit
INIT_SOC_FRAC    = 0.5   # start each day at 50%

# Grid / market params (match _get_grid_prices ToU schedule)
RETAIL_PRICE_OFF = 0.20  # $/kWh  (off-peak)
RETAIL_PRICE_PK  = 0.50  # $/kWh  (17-21h peak)
FEED_IN_TARIFF   = 0.10  # $/kWh  (constant)
P2P_PRICE_OFF    = (RETAIL_PRICE_OFF + FEED_IN_TARIFF) / 2   # midpoint
P2P_PRICE_PK     = (RETAIL_PRICE_PK  + FEED_IN_TARIFF) / 2

MAX_LINE_KW      = 50.0  # physical line capacity
CHURN_COST       = 0.0001  # $/kWh — prevents LP from churning battery

# ── normalization scale factors from normalization_config.json ───────────────
NORM_CFG_PATH = os.path.join(PROJECT_ROOT, "normalization_config.json")
if os.path.exists(NORM_CFG_PATH):
    with open(NORM_CFG_PATH) as f:
        _norm = json.load(f)
else:
    _norm = {}

# Scale factors: multiply normalised CSV [0,1] → real physical units
SCALE = {
    "agent_0_demand": _norm.get("agent_0_demand_max", 1.0),
    "agent_0_gen":    0.0,   # Agent 0 is pure consumer
    "agent_1_demand": _norm.get("agent_1_demand_max", 1.0),
    "agent_1_gen":    _norm.get("agent_1_wind_max",   2.937),
    "agent_2_demand": _norm.get("agent_2_demand_max", 1.0),
    "agent_2_gen":    _norm.get("agent_2_pv_max",     1.0),
    "agent_3_demand": _norm.get("agent_3_demand_max", 1.0),
    "agent_3_gen":    _norm.get("agent_3_pv_max",     1.0),
}


# ─────────────────────────────── Data helpers ────────────────────────────────
def load_and_denormalise(data_path: str) -> pd.DataFrame:
    """Read CSV, parse timestamps, check hourly continuity, denormalise."""
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure hourly frequency; interpolate any gaps
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_range).interpolate(method="time").reset_index()
    df.rename(columns={"index": "timestamp"}, inplace=True)

    # ── un-normalise the physical columns ────────────────────────────────────
    col_map = {
        "agent_1_demand": ("agent_1_demand", SCALE["agent_1_demand"]),
        "agent_1_wind":   ("agent_1_gen",    SCALE["agent_1_gen"]),
        "agent_2_demand": ("agent_2_demand", SCALE["agent_2_demand"]),
        "agent_2_pv":     ("agent_2_gen",    SCALE["agent_2_gen"]),
        "agent_3_demand": ("agent_3_demand", SCALE["agent_3_demand"]),
        "agent_3_pv":     ("agent_3_gen",    SCALE["agent_3_gen"]),
    }
    for csv_col, (new_col, factor) in col_map.items():
        if csv_col in df.columns:
            df[new_col] = df[csv_col].fillna(0.0) * factor
        else:
            df[new_col] = 0.0

    # Agent 0 — pure consumer; no generation data
    df["agent_0_demand"] = 0.0   # no separate consumer column → set to 0 kW
    df["agent_0_gen"]    = 0.0

    df["date"] = df["timestamp"].dt.date
    return df


def get_price_schedule():
    """Returns (price_buy[24], price_sell[24], p2p_price[24]) arrays."""
    price_buy  = np.full(HORIZON, RETAIL_PRICE_OFF)
    price_sell = np.full(HORIZON, FEED_IN_TARIFF)
    p2p_price  = np.full(HORIZON, P2P_PRICE_OFF)
    price_buy[17:21]  = RETAIL_PRICE_PK
    p2p_price[17:21]  = P2P_PRICE_PK
    return price_buy, price_sell, p2p_price


# ─────────────────────────────── OPF solver ──────────────────────────────────
def solve_day_opf(demands: np.ndarray,
                  generations: np.ndarray,
                  price_buy: np.ndarray,
                  price_sell: np.ndarray,
                  p2p_price: np.ndarray,
                  verbose: bool = False) -> dict:
    """
    Solve 24-hour OPF for N_AGENTS using cvxpy.

    Parameters
    ----------
    demands      : (HORIZON, N_AGENTS)  real kW
    generations  : (HORIZON, N_AGENTS)  real kW
    price_buy    : (HORIZON,)           $/kWh retail buy
    price_sell   : (HORIZON,)           $/kWh feed-in tariff
    p2p_price    : (HORIZON,)           $/kWh P2P clearing price (midpoint)

    Returns
    -------
    dict with solver status, total cost, per-timestep arrays, etc.
    """
    T  = HORIZON
    N  = N_AGENTS

    # ── Decision variables (all non-negative) ────────────────────────────────
    p_charge    = cp.Variable((N, T), nonneg=True, name="p_charge")    # kW
    p_discharge = cp.Variable((N, T), nonneg=True, name="p_discharge") # kW
    soc         = cp.Variable((N, T + 1), nonneg=True, name="soc")     # kWh
    p2p_sell    = cp.Variable((N, T), nonneg=True, name="p2p_sell")    # kW
    p2p_buy     = cp.Variable((N, T), nonneg=True, name="p2p_buy")     # kW
    grid_imp    = cp.Variable((N, T), nonneg=True, name="grid_imp")    # kW
    grid_exp    = cp.Variable((N, T), nonneg=True, name="grid_exp")    # kW

    constraints = []

    for i in range(N):
        base_cap = BATTERY_CAP_KWH[i]
        p_max    = MAX_POWER_KW[i]
        soc_init = base_cap * INIT_SOC_FRAC

        # ── Initial SoC ──────────────────────────────────────────────────────
        constraints.append(soc[i, 0] == soc_init)

        for t in range(T):
            # Dynamic capacity: EV (agent 2) is "away" 08:00–17:00
            if i == 2 and 8 <= t < 17:
                cap_t = 2.0
            else:
                cap_t = base_cap

            # SoC bounds
            constraints.append(soc[i, t]     >= 0.0)
            constraints.append(soc[i, t]     <= cap_t)
            constraints.append(soc[i, t + 1] >= 0.0)
            constraints.append(soc[i, t + 1] <= cap_t)

            # Battery power limits
            constraints.append(p_charge[i, t]    <= p_max)
            constraints.append(p_discharge[i, t] <= p_max)

            # SoC dynamics (split-efficiency)
            constraints.append(
                soc[i, t + 1] == soc[i, t]
                + EFF_CHARGE * p_charge[i, t]
                - p_discharge[i, t] / EFF_DISCH
            )

            # Power balance per agent per timestep
            # gen + discharge + p2p_buy + grid_imp == demand + charge + p2p_sell + grid_exp
            constraints.append(
                generations[t, i] + p_discharge[i, t] + p2p_buy[i, t] + grid_imp[i, t]
                == demands[t, i] + p_charge[i, t] + p2p_sell[i, t] + grid_exp[i, t]
            )

        # ── Terminal SoC constraint: end-of-day ≥ start-of-day ───────────────
        # Prevents "emptying the battery for free profit" at end of horizon
        constraints.append(soc[i, T] >= soc_init)
        constraints.append(soc[i, T] <= base_cap)

    # ── P2P market balance per timestep: Σ_i (sell - buy) == 0 ─────────────
    for t in range(T):
        constraints.append(cp.sum(p2p_sell[:, t] - p2p_buy[:, t]) == 0)

    # ── Physical line capacity ────────────────────────────────────────────────
    for t in range(T):
        constraints.append(cp.sum(grid_imp[:, t]) <= MAX_LINE_KW)
        constraints.append(cp.sum(grid_exp[:, t]) <= MAX_LINE_KW)

    # ── Objective: Minimise total community electricity cost ─────────────────
    # cost = Σ_t Σ_i (grid_imp * price_buy - grid_exp * price_sell)
    # minus P2P revenue (sellers earn P2P price, buyers save retail price)
    # Battery churn prevention: tiny running cost on charge+discharge
    cost_terms = []
    for t in range(T):
        pb  = price_buy[t]
        ps  = price_sell[t]
        pp  = p2p_price[t]
        for i in range(N):
            cost_terms.append(grid_imp[i, t] * pb)        # pay retail for import
            cost_terms.append(-grid_exp[i, t] * ps)       # earn feed-in for export
            cost_terms.append(-p2p_sell[i, t] * pp)       # earn P2P price for sell
            cost_terms.append(p2p_buy[i, t] * pp)         # pay P2P price for buy
            # Churn cost: discourages simultaneous charge+discharge
            cost_terms.append(CHURN_COST * p_charge[i, t])
            cost_terms.append(CHURN_COST * p_discharge[i, t])

    objective = cp.Minimize(cp.sum(cost_terms))
    problem   = cp.Problem(objective, constraints)

    # ── Solve: prefer CLARABEL, fall back to ECOS ────────────────────────────
    status = "failed"
    total_cost = None
    for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=verbose)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                status     = problem.status
                total_cost = float(problem.value)
                break
        except Exception as e:
            if verbose:
                print(f"  Solver {solver} failed: {e}")
            continue

    if total_cost is None:
        return {"status": "failed", "total_cost": None,
                "p2p_volume_kwh": 0.0, "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0}

    p2p_vol  = float(cp.sum(p2p_sell).value) if p2p_sell.value is not None else 0.0
    g_imp    = float(cp.sum(grid_imp).value) if grid_imp.value is not None else 0.0
    g_exp    = float(cp.sum(grid_exp).value) if grid_exp.value is not None else 0.0

    return {
        "status":          status,
        "total_cost":      total_cost,  # negative = profit
        "p2p_volume_kwh":  p2p_vol,
        "grid_import_kwh": g_imp,
        "grid_export_kwh": g_exp,
    }


# ──────────────────────────────── Main loop ──────────────────────────────────
def run_mpc_baseline(data_path: str = "processed_hybrid_data.csv",
                     max_days: int = None,
                     verbose: bool = False):
    """
    Run MPC OPF across all (or max_days) days in the dataset.
    Prints a per-day table and writes research_q1/results/mpc_results.csv.
    """
    print("\n" + "=" * 65)
    print("  PHASE D — MPC 'Gold Standard' OPF Baseline")
    print("  Theoretical Maximum Profit / Minimum Cost Oracle")
    print("=" * 65)

    # ── Load & denormalise ───────────────────────────────────────────────────
    full_path = data_path if os.path.isabs(data_path) else os.path.join(PROJECT_ROOT, data_path)
    print(f"\nLoading: {full_path}")
    df = load_and_denormalise(full_path)

    unique_dates = sorted(df["date"].unique())
    if max_days is not None:
        unique_dates = unique_dates[:max_days]
    print(f"Solving OPF for {len(unique_dates)} day(s) ...")

    price_buy, price_sell, p2p_price = get_price_schedule()

    records = []
    total_community_cost = 0.0

    for day_idx, date in enumerate(unique_dates):
        day_df = df[df["date"] == date].reset_index(drop=True)

        # Pad/trim to exactly 24 rows
        if len(day_df) < HORIZON:
            day_df = day_df.reindex(range(HORIZON)).fillna(0.0)
        elif len(day_df) > HORIZON:
            day_df = day_df.iloc[:HORIZON]

        # Build demand / generation matrices (T, N)
        demands     = np.zeros((HORIZON, N_AGENTS))
        generations = np.zeros((HORIZON, N_AGENTS))

        for i in range(N_AGENTS):
            dem_col = f"agent_{i}_demand"
            gen_col = f"agent_{i}_gen"
            if dem_col in day_df.columns:
                demands[:, i] = day_df[dem_col].fillna(0.0).values
            if gen_col in day_df.columns:
                generations[:, i] = day_df[gen_col].fillna(0.0).values

        result = solve_day_opf(demands, generations,
                               price_buy, price_sell, p2p_price,
                               verbose=verbose)

        cost = result["total_cost"] if result["total_cost"] is not None else float("nan")
        if result["status"] not in ("failed",):
            total_community_cost += cost

        profit_str = f"${-cost:+.4f}" if not np.isnan(cost) else "  N/A  "
        print(f"Day {day_idx+1:3d} | {date} | Status: {result['status']:20s} | "
              f"Profit: {profit_str} | P2P vol: {result['p2p_volume_kwh']:.2f} kWh")

        records.append({
            "date":               str(date),
            "solver_status":      result["status"],
            "total_cost_$":       round(cost, 4)  if not np.isnan(cost) else None,
            "profit_$":           round(-cost, 4) if not np.isnan(cost) else None,
            "p2p_volume_kwh":     round(result["p2p_volume_kwh"], 2),
            "grid_import_kwh":    round(result["grid_import_kwh"], 2),
            "grid_export_kwh":    round(result["grid_export_kwh"], 2),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total_profit = -total_community_cost
    print("\n" + "─" * 65)
    print(f"  THEORETICAL MAXIMUM PROFIT over {len(unique_dates)} day(s):")
    print(f"    Total Community Profit   : ${total_profit:+.2f}")
    print(f"    Avg Profit / Day         : ${total_profit/max(len(unique_dates),1):+.2f}")
    print(f"    Avg Profit / Agent / Day : ${total_profit/max(len(unique_dates)*N_AGENTS,1):+.2f}")
    print("─" * 65)
    print("\n  ┌─────────────────────────── Comparison Guide ───────────────────┐")
    print("  │  MPC result = theoretical ceiling (oracle with full knowledge)  │")
    print("  │  Heuristic  = rule-based upper bound without battery look-ahead │")
    print("  │  RL Agent   = learned policy goal: approach MPC line closely    │")
    print("  └────────────────────────────────────────────────────────────────┘\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, "research_q1", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "mpc_results.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Results saved → {out_path}\n")

    return records, total_profit


# ─────────────────────────────── CLI entry ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase D: MPC Gold Standard Baseline")
    parser.add_argument(
        "--data-path", type=str, default="processed_hybrid_data.csv",
        help="Path to processed_hybrid_data.csv (relative to project root or absolute)"
    )
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Limit number of days to solve (default: all days in dataset)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print cvxpy solver output"
    )
    args = parser.parse_args()

    run_mpc_baseline(
        data_path=args.data_path,
        max_days=args.max_days,
        verbose=args.verbose,
    )
