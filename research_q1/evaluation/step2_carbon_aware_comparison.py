"""
step2_carbon_aware_comparison.py
=================================
Validates carbon-aware behavior by comparing:
  A) Standard model (CO2_PENAL_COEFF = 0.10, as trained)
  B) Ablated model  (CO2_PENAL_COEFF = 0.0, same weights, patched env)

Both evaluations use the SAME trained PPO model and SAME start conditions.
The only difference is whether the carbon penalty is active in the env reward.

CO2 emissions are computed physically (not from reward):
  co2[t] = grid_import[t] * timestep_hours * CARBON_INTENSITY

Clean profit uses the corrected definition from Step 1:
  clean_profit[t] = trading_revenue[t] - grid_cost[t]

Outputs:
  research_q1/results/step2_comparison_table.csv
  research_q1/results/step2_co2_comparison.png
  research_q1/results/step2_profit_comparison.png
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

N_AGENTS           = 4
TIMESTEP_HOURS     = 1.0
RETAIL_OFF_PEAK    = 0.20
RETAIL_PEAK        = 0.50
PEAK_HOURS         = set(range(17, 21))
CARBON_INTENSITY   = 0.233
MODEL_PATH         = os.path.join(ROOT, "models_slim", "seed_0", "best_model.zip")
DATA_PATH          = os.path.join(ROOT, "processed_hybrid_data.csv")
N_EPISODES         = 3
SEED               = 42


def retail_price(step_in_episode: int) -> float:
    return RETAIL_PEAK if (step_in_episode % 24) in PEAK_HOURS else RETAIL_OFF_PEAK


# ── Patched env: zeroes out CO2 penalty by overriding one constant ────────────
class NoCO2PenaltyEnv:
    """
    Thin wrapper that patches CO2_PENAL_COEFF to 0.0 while keeping all
    other environment logic identical. Uses __getattr__ delegation.
    """

    def __init__(self, base_env):
        self._env = base_env
        # Store original method; we will monkey-patch before each step call
        import types
        self._original_step_grid = base_env._step_grid_and_reward.__func__

        def patched_step_grid(inner_self, physics_state, market_results):
            # Call original but capture result
            obs, reward, info = self._original_step_grid(inner_self, physics_state, market_results)

            # Add back the CO2 penalty term that was subtracted
            # info has 'mean_co2_penalty' from reward_tracker.get_info()
            co2_penalty_refund = inner_self.reward_tracker.step_co2_penalty.sum()
            reward = reward + co2_penalty_refund

            # Zero the stored component so it does not appear in future breakdowns
            inner_self.reward_tracker.step_co2_penalty[:] = 0.0

            return obs, reward, info

        self._env._step_grid_and_reward = types.MethodType(patched_step_grid, self._env)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_base_env(data_path: str, disable_co2: bool = False):
    from train.energy_env_robust import EnergyMarketEnvRobust
    env = EnergyMarketEnvRobust(
        n_agents=N_AGENTS, data_file=data_path, random_start_day=False,
        enable_ramp_rates=True, enable_losses=True, forecast_horizon=4,
        enable_predictive_obs=True, forecast_noise_std=0.0, diversity_mode=True,
    )
    if disable_co2:
        env = NoCO2PenaltyEnv(env)
    return env


def run_evaluation(model, config_label: str, disable_co2: bool,
                   n_episodes: int, seed: int, data_path: str) -> pd.DataFrame:
    rows = []
    for ep in range(n_episodes):
        env = make_base_env(data_path, disable_co2=disable_co2)
        obs, _ = env.reset(seed=seed + ep)
        done, step = False, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trading_revenue = info.get("mean_profit", 0.0) * N_AGENTS
            r_price         = retail_price(step)
            grid_cost       = info.get("total_import", 0.0) * TIMESTEP_HOURS * r_price
            clean_profit    = trading_revenue - grid_cost
            co2_kg          = info.get("total_import", 0.0) * TIMESTEP_HOURS * CARBON_INTENSITY
            grid_import_kw  = float(info.get("total_import", 0.0))

            rows.append({
                "config":           config_label,
                "episode":          ep,
                "step":             step,
                "clean_profit":     clean_profit,
                "trading_revenue":  trading_revenue,
                "grid_cost":        grid_cost,
                "co2_kg":           co2_kg,
                "grid_import_kw":   grid_import_kw,
                "p2p_volume_kwh":   float(info.get("p2p_volume_kwh_step", 0.0)),
                "market_price":     float(info.get("market_price", 0.0)),
            })
            step += 1

    return pd.DataFrame(rows)


def build_comparison_table(df_co2, df_no_co2) -> pd.DataFrame:
    def summarize(df, label):
        return {
            "config":                label,
            "mean_step_profit_usd":  df["clean_profit"].mean(),
            "cum_profit_usd":        df.groupby("episode")["clean_profit"].sum().mean(),
            "mean_co2_kg_step":      df["co2_kg"].mean(),
            "cum_co2_kg":            df.groupby("episode")["co2_kg"].sum().mean(),
            "mean_grid_import_kw":   df["grid_import_kw"].mean(),
            "p2p_volume_kwh_total":  df["p2p_volume_kwh"].sum(),
        }

    table = pd.DataFrame([
        summarize(df_co2,    "carbon_aware   (CO2_coeff=0.10)"),
        summarize(df_no_co2, "no_co2_penalty (CO2_coeff=0.00)"),
    ])

    # Compute relative changes
    pct_co2_change    = ((table.loc[1, "cum_co2_kg"] - table.loc[0, "cum_co2_kg"])
                         / (abs(table.loc[0, "cum_co2_kg"]) + 1e-9) * 100)
    pct_profit_change = ((table.loc[1, "cum_profit_usd"] - table.loc[0, "cum_profit_usd"])
                         / (abs(table.loc[0, "cum_profit_usd"]) + 1e-9) * 100)

    print("\n" + "="*65)
    print("  STEP 2: Carbon-Aware vs No-CO2-Penalty Comparison")
    print("="*65)
    print(f"  {'Metric':<32}  {'Carbon-Aware':>14}  {'No CO2':>12}")
    print(f"  {'-'*32}  {'-'*14}  {'-'*12}")
    for col in ["cum_profit_usd", "cum_co2_kg", "mean_grid_import_kw", "p2p_volume_kwh_total"]:
        v0 = table.loc[0, col]
        v1 = table.loc[1, col]
        print(f"  {col:<32}  {v0:>14.4f}  {v1:>12.4f}")
    print(f"\n  CO2 change  (no_co2 vs carbon_aware): {pct_co2_change:+.2f}%")
    print(f"  Profit change                        : {pct_profit_change:+.2f}%")

    if pct_co2_change > 0:
        print("  [PASS] Carbon-aware model reduces CO2 vs no-penalty baseline.")
    else:
        print("  [WARN] Carbon-aware model does NOT reduce CO2 vs baseline.")
        print("         This may indicate policy collapse or minimal grid import.")
    print("="*65)
    return table


def make_comparison_plots(df_std, df_no_co2, out_dir: str):
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("ggplot")

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    avg_std    = df_std.groupby("step")[["clean_profit", "co2_kg"]].mean().reset_index()
    avg_no_co2 = df_no_co2.groupby("step")[["clean_profit", "co2_kg"]].mean().reset_index()
    hours      = avg_std["step"].values

    # ── Figure A: Cumulative CO2 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hours, avg_std["co2_kg"].cumsum(),    lw=2, color=colors[0],
            label="Carbon-Aware (CO2 penalty ON)")
    ax.plot(hours, avg_no_co2["co2_kg"].cumsum(), lw=2, color=colors[1], linestyle="--",
            label="No CO2 Penalty (ablated)")
    ax.fill_between(
        hours,
        avg_std["co2_kg"].cumsum(),
        avg_no_co2["co2_kg"].cumsum(),
        alpha=0.10, color=colors[0],
        label="Emission reduction")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative CO2 Emissions (kg)")
    ax.set_title("Step 2: Carbon-Aware vs Baseline — Cumulative CO2")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    p = os.path.join(out_dir, "step2_co2_comparison.png")
    plt.savefig(p); plt.savefig(p.replace(".png", ".pdf")); plt.close()
    print(f"  Saved -> {p}")

    # ── Figure B: Cumulative Clean Profit ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hours, avg_std["clean_profit"].cumsum(),    lw=2, color=colors[0],
            label="Carbon-Aware")
    ax.plot(hours, avg_no_co2["clean_profit"].cumsum(), lw=2, color=colors[1], linestyle="--",
            label="No CO2 Penalty")
    ax.axhline(0, color="grey", lw=0.6, linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Clean Profit (USD)")
    ax.set_title("Step 2: Carbon-Aware vs Baseline — Cumulative Clean Profit")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    p = os.path.join(out_dir, "step2_profit_comparison.png")
    plt.savefig(p); plt.savefig(p.replace(".png", ".pdf")); plt.close()
    print(f"  Saved -> {p}")


def main():
    from stable_baselines3 import PPO

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found:", MODEL_PATH)
        return

    print("[STEP 2] Loading model:", MODEL_PATH)
    model = PPO.load(MODEL_PATH)

    print("  Running carbon-aware evaluation ...")
    df_std = run_evaluation(model, "carbon_aware", disable_co2=False,
                            n_episodes=N_EPISODES, seed=SEED, data_path=DATA_PATH)

    print("  Running no-CO2-penalty evaluation ...")
    df_no = run_evaluation(model, "no_co2_penalty", disable_co2=True,
                           n_episodes=N_EPISODES, seed=SEED, data_path=DATA_PATH)

    table = build_comparison_table(df_std, df_no)

    csv_path = os.path.join(OUT_DIR, "step2_comparison_table.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n[INFO] Comparison table saved -> {csv_path}")

    all_df = pd.concat([df_std, df_no], ignore_index=True)
    raw_csv = os.path.join(OUT_DIR, "step2_step_data.csv")
    all_df.to_csv(raw_csv, index=False)
    print(f"[INFO] Step-level data saved -> {raw_csv}")

    make_comparison_plots(df_std, df_no, OUT_DIR)
    print("[STEP 2] COMPLETE\n")


if __name__ == "__main__":
    main()
