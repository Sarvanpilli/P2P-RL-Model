"""
step1_profit_diagnostics.py
============================
CORRECTED economic profit definition.

  clean_profit[t] = trading_revenue[t] - grid_cost[t]

Where:
  trading_revenue[t] = sum_agents(trades[t] * clearing_price[t])
                     = info['mean_profit'] * N_AGENTS
  grid_cost[t]       = info['total_import'] * timestep_hours * retail_price[t]
                       (retail_price = 0.50 during peaks 17-21h, else 0.20)

Intentionally EXCLUDED from clean_profit:
  - CO2 penalty          (reward shaping, not economic)
  - Fairness penalty     (reward shaping, not economic)
  - Lagrangian penalty   (constraint enforcement, not economic)
  - Grid import penalty  (reward shaping duplicate of grid_cost)
  - P2P bonus            (reward shaping, not economic revenue)

Battery degradation is tracked separately and flagged in debug output.

Outputs:
  research_q1/results/step1_profit_components.csv
  research_q1/results/step1_profit_vs_reward.png
  research_q1/results/step1_reward_decomposition.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "research_q1", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_AGENTS           = 4
TIMESTEP_HOURS     = 1.0
RETAIL_OFF_PEAK    = 0.20   # $/kWh
RETAIL_PEAK        = 0.50   # $/kWh, hours 17–21
PEAK_HOURS         = set(range(17, 21))
CARBON_INTENSITY   = 0.233  # kg CO2/kWh (grid average)
MODEL_PATH         = os.path.join(ROOT, "models_slim", "seed_0", "best_model.zip")
DATA_PATH          = os.path.join(ROOT, "processed_hybrid_data.csv")
N_EPISODES         = 3
SEED               = 42


def retail_price(step_in_episode: int) -> float:
    h = step_in_episode % 24
    return RETAIL_PEAK if h in PEAK_HOURS else RETAIL_OFF_PEAK


def evaluate_episode(model, ep_idx: int, seed: int, data_path: str) -> pd.DataFrame:
    from train.energy_env_robust import EnergyMarketEnvRobust

    env = EnergyMarketEnvRobust(
        n_agents=N_AGENTS, data_file=data_path, random_start_day=False,
        enable_ramp_rates=True, enable_losses=True, forecast_horizon=4,
        enable_predictive_obs=True, forecast_noise_std=0.0, diversity_mode=True,
    )
    obs, _ = env.reset(seed=seed + ep_idx)
    done, step, rows = False, 0, []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, total_rl_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── INFO DICT COMPONENTS ───────────────────────────────────────────────
        # Raw trading P&L: sum of (trades_i * clearing_price) across all agents
        trading_revenue = info.get("mean_profit", 0.0) * N_AGENTS

        # Actual economic cost of importing from grid (NOT reward shaping)
        r_price = retail_price(step)
        grid_cost = info.get("total_import", 0.0) * TIMESTEP_HOURS * r_price

        # ── CLEAN ECONOMIC PROFIT ──────────────────────────────────────────────
        clean_profit = trading_revenue - grid_cost

        # ── REWARD SHAPING COMPONENTS (for diagnostics only) ──────────────────
        co2_penalty      = info.get("mean_co2_penalty", 0.0) * N_AGENTS
        battery_cost     = info.get("mean_battery_cost", 0.0) * N_AGENTS
        fairness_penalty = info.get("fairness_penalty", 0.0)
        lag_penalty      = info.get("lagrangian/penalty_this_step", 0.0)
        grid_penalty_rs  = info.get("mean_grid_penalty", 0.0) * N_AGENTS  # reward shaping
        p2p_bonus        = info.get("mean_p2p_bonus", 0.0) * N_AGENTS

        # ── CO2 EMISSIONS (physical) ───────────────────────────────────────────
        co2_kg = info.get("total_import", 0.0) * TIMESTEP_HOURS * CARBON_INTENSITY

        rows.append({
            "episode":           ep_idx,
            "step":              step,
            "hour":              step % 24,
            "retail_price":      r_price,
            # Economic metrics (clean)
            "trading_revenue":   trading_revenue,
            "grid_cost":         grid_cost,
            "clean_profit":      clean_profit,
            # Reward shaping terms (separate)
            "co2_penalty":       co2_penalty,
            "battery_cost":      battery_cost,
            "fairness_penalty":  fairness_penalty,
            "lag_penalty":       lag_penalty,
            "grid_penalty_rs":   grid_penalty_rs,
            "p2p_bonus":         p2p_bonus,
            # Total RL reward (includes all shaping)
            "total_rl_reward":   float(total_rl_reward),
            # Physical metrics
            "total_import_kw":   float(info.get("total_import", 0.0)),
            "p2p_volume_kwh":    float(info.get("p2p_volume_kwh_step", 0.0)),
            "market_price":      float(info.get("market_price", 0.0)),
            "co2_kg":            co2_kg,
        })
        step += 1

    return pd.DataFrame(rows)


def print_component_table(df: pd.DataFrame):
    means = df[[
        "trading_revenue", "grid_cost", "clean_profit",
        "co2_penalty", "battery_cost", "fairness_penalty",
        "lag_penalty", "grid_penalty_rs", "p2p_bonus",
        "total_rl_reward"
    ]].mean()

    print("\n" + "="*60)
    print("  STEP 1: Reward Component Breakdown (mean per step)")
    print("="*60)
    print(f"  [ECONOMIC - clean_profit components]")
    print(f"    trading_revenue    : {means['trading_revenue']:+.4f} USD")
    print(f"    grid_cost          : {means['grid_cost']:+.4f} USD  (deducted)")
    print(f"    ----------------------------------------")
    print(f"    CLEAN PROFIT       : {means['clean_profit']:+.4f} USD  <-- use this for paper")
    print(f"\n  [REWARD SHAPING - excluded from clean_profit]")
    print(f"    co2_penalty        : {means['co2_penalty']:+.4f}")
    print(f"    battery_cost       : {means['battery_cost']:+.4f}")
    print(f"    fairness_penalty   : {means['fairness_penalty']:+.4f}")
    print(f"    lagrangian_penalty : {means['lag_penalty']:+.4f}")
    print(f"    grid_penalty_rs    : {means['grid_penalty_rs']:+.4f}")
    print(f"    p2p_bonus          : {means['p2p_bonus']:+.4f}")
    print(f"\n  [TOTAL RL REWARD (all terms combined)]")
    print(f"    total_rl_reward    : {means['total_rl_reward']:+.4f}")
    print("="*60)

    # Validation check
    penalty_sum = (means['co2_penalty'] + means['battery_cost'] +
                   means['fairness_penalty'] + means['lag_penalty'] +
                   means['grid_penalty_rs'])
    inflation_pct = abs(penalty_sum / (means['clean_profit'] + 1e-8)) * 100
    print(f"\n  [CHECK] Penalty inflation vs clean_profit: {inflation_pct:.1f}%")
    if inflation_pct > 200:
        print("  [WARN] Penalties dominate reward signal (>200% of clean profit).")
        print("         This confirms reward shaping is distorting economic signal.")
    else:
        print("  [PASS] Penalty inflation within acceptable range.")


def make_profit_vs_reward_plot(df: pd.DataFrame, out_dir: str):
    """Two-line plot: cumulative clean_profit vs cumulative total_rl_reward."""
    avg = df.groupby("step")[["clean_profit", "total_rl_reward"]].mean().reset_index()
    hours = avg["step"].values
    cum_profit = avg["clean_profit"].cumsum().values
    cum_reward = avg["total_rl_reward"].cumsum().values

    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("ggplot")

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })

    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hours, cum_profit, lw=2.0, color=colors[0],
            label="Clean Profit (trading_revenue - grid_cost)")
    ax.plot(hours, cum_reward, lw=2.0, color=colors[1], linestyle="--",
            label="Total RL Reward (includes penalty shaping)")
    ax.axhline(0, color="grey", lw=0.6, linestyle=":", alpha=0.7)
    ax.fill_between(hours, cum_profit, cum_reward, alpha=0.08, color="grey",
                    label="Shaping distortion")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Value (USD)")
    ax.set_title("Step 1: Clean Economic Profit vs Total RL Reward")
    ax.legend(loc="lower left")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    fname = os.path.join(out_dir, "step1_profit_vs_reward.png")
    plt.savefig(fname)
    plt.savefig(fname.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved -> {fname}")


def make_decomposition_plot(df: pd.DataFrame, out_dir: str):
    """Stacked-area: each shaping component's per-step magnitude over time."""
    avg = df.groupby("step")[[
        "trading_revenue", "co2_penalty", "battery_cost",
        "fairness_penalty", "lag_penalty", "p2p_bonus"
    ]].mean().reset_index()
    hours = avg["step"].values

    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("ggplot")

    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top panel: clean profit components
    ax = axes[0]
    ax.plot(hours, avg["trading_revenue"], color=colors[0], lw=1.6,
            label="Trading Revenue")
    ax.fill_between(hours, avg["trading_revenue"], 0, alpha=0.10, color=colors[0])
    ax.axhline(0, color="grey", lw=0.5, linestyle=":")
    ax.set_ylabel("USD / step")
    ax.set_title("Economic Components (Revenue vs Grid Cost)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom panel: shaping terms (all plotted as positive magnitudes)
    ax2 = axes[1]
    shaping_terms = {
        "CO2 Penalty":     avg["co2_penalty"].abs(),
        "Battery Cost":    avg["battery_cost"].abs(),
        "Fairness Penalty":avg["fairness_penalty"].abs(),
        "Lagrangian":      avg["lag_penalty"].abs(),
        "P2P Bonus":       avg["p2p_bonus"].abs(),
    }
    bottom = np.zeros(len(hours))
    for (label, vals), color in zip(shaping_terms.items(), colors[1:]):
        ax2.bar(hours, vals.values, bottom=bottom, label=label,
                color=color, alpha=0.75, width=1.0)
        bottom += vals.values

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Magnitude (USD / step)")
    ax2.set_title("Reward Shaping Terms (excluded from clean profit)")
    ax2.legend(fontsize=9, ncol=2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, "step1_reward_decomposition.png")
    plt.savefig(fname)
    plt.savefig(fname.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved -> {fname}")


def main():
    from stable_baselines3 import PPO

    print("[STEP 1] Loading model:", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found:", MODEL_PATH)
        return

    model = PPO.load(MODEL_PATH)
    all_dfs = []
    for ep in range(N_EPISODES):
        ep_df = evaluate_episode(model, ep, SEED, DATA_PATH)
        all_dfs.append(ep_df)
        total_clean = ep_df["clean_profit"].sum()
        total_grid  = ep_df["grid_cost"].sum()
        total_rev   = ep_df["trading_revenue"].sum()
        steps       = len(ep_df)
        print(f"  Ep {ep+1}/{N_EPISODES}  steps={steps}"
              f"  revenue={total_rev:.2f}  grid_cost={total_grid:.2f}"
              f"  clean_profit={total_clean:.2f}")

    df = pd.concat(all_dfs, ignore_index=True)
    avg_df = df.groupby("step").mean().reset_index()

    csv_path = os.path.join(OUT_DIR, "step1_profit_components.csv")
    avg_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Per-step averages saved -> {csv_path}")

    print_component_table(df)
    make_profit_vs_reward_plot(avg_df, OUT_DIR)
    make_decomposition_plot(avg_df, OUT_DIR)

    total_clean_profit = avg_df["clean_profit"].sum()
    total_co2          = avg_df["co2_kg"].sum()
    print(f"\n[SUMMARY Step 1]")
    print(f"  Episodes evaluated   : {N_EPISODES}")
    print(f"  Steps per episode    : {len(avg_df)}")
    print(f"  Cumulative clean profit : {total_clean_profit:.3f} USD")
    print(f"  Cumulative CO2          : {total_co2:.3f} kg")
    print(f"  P2P volume (total)      : {avg_df['p2p_volume_kwh'].sum():.4f} kWh")
    if avg_df["p2p_volume_kwh"].sum() < 1e-3:
        print("  [FIND] P2P volume near zero -> Nash equilibrium collapse detected.")
        print("         Agents are not engaging in peer-to-peer trades.")
    print("[STEP 1] COMPLETE\n")


if __name__ == "__main__":
    main()
