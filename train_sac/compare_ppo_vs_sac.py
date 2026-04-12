"""
compare_ppo_vs_sac.py
======================
Generate a side-by-side comparative analysis of PPO (SLIM v2) vs SAC
on the P2P Energy Trading environment.

This script:
1. Evaluates the PPO best model (models_slim/seed_0/best_model.zip)
2. Evaluates the SAC best model (models_sac/best/best_model.zip)
3. Produces detailed metrics table + 4 comparison plots

Usage:
    # After SAC training is complete:
    python train_sac/compare_ppo_vs_sac.py

    # With custom paths:
    python train_sac/compare_ppo_vs_sac.py \\
        --ppo_model models_slim/seed_0/best_model.zip \\
        --sac_model models_sac/best/best_model.zip

Outputs (in train_sac/results/):
    comparison_metrics.csv     - per-step metrics for BOTH algorithms
    comparison_summary.csv     - aggregated KPIs (mean ± std)
    plot_reward.png            - episode reward convergence
    plot_p2p_volume.png        - P2P trading volume per step
    plot_grid_flow.png         - grid import vs export
    plot_market_price.png      - market clearing price over time
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train.energy_env_robust import EnergyMarketEnvRobust

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─────────────────────────────────────────────────────────────────────────────
# Shared Env factory
# ─────────────────────────────────────────────────────────────────────────────
def make_raw_env(data_file: str) -> EnergyMarketEnvRobust:
    return EnergyMarketEnvRobust(
        n_agents=4,
        data_file=data_file,
        random_start_day=False,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4,
        enable_predictive_obs=True,
        forecast_noise_std=0.0,
        diversity_mode=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    label: str,
    model,
    data_file: str,
    n_episodes: int,
    seed: int,
    vec_normalize_path: str = None,
) -> pd.DataFrame:
    """Run model for n_episodes; collect step metrics."""
    rows = []
    for ep in range(n_episodes):
        env = make_raw_env(data_file)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        ep_reward = 0.0

        while not done:
            # PPO may need a VecEnv wrapper; SAC works on raw obs
            # For both we just use model.predict which accepts np arrays
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rows.append({
                "episode": ep,
                "step": step,
                "algorithm": label,
                "reward": float(reward),
                "p2p_volume_kwh": float(info.get("p2p_volume_kwh_step", 0.0)),
                "total_export_kw": float(info.get("total_export", 0.0)),
                "total_import_kw": float(info.get("total_import", 0.0)),
                "market_price": float(info.get("market_price", 0.0)),
                "line_overload_kw": float(info.get("line_overload_kw", 0.0)),
                "failed_trades": float(info.get("failed_trades", 0)),
                "soc_violation": float(info.get("lagrangian/violation_soc", 0.0)),
                "absolute_profit_usd": float(info.get("absolute_profit_usd", 0.0)),
            })
            ep_reward += float(reward)
            step += 1

        print(f"  [{label}] Episode {ep+1}/{n_episodes} | Steps: {step} | Reward: {ep_reward:.2f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"PPO": "#4CAF50", "SAC": "#2196F3"}
STYLE  = {"PPO": "-",       "SAC": "--"}

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_over_steps(df: pd.DataFrame, metric: str, title: str,
                            ylabel: str, filename: str):
    """Per-step line plot averaged across episodes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for algo, grp in df.groupby("algorithm"):
        pivot = grp.groupby("step")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(pivot["step"], pivot["mean"],
                label=algo, color=COLORS.get(algo, "gray"),
                linestyle=STYLE.get(algo, "-"), linewidth=1.8)
        ax.fill_between(pivot["step"],
                        pivot["mean"] - pivot["std"],
                        pivot["mean"] + pivot["std"],
                        alpha=0.15, color=COLORS.get(algo, "gray"))
    _style_ax(ax, title, "Time Step", ylabel)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved plot → {out}")


def plot_episode_totals(df: pd.DataFrame, metric: str, title: str,
                         ylabel: str, filename: str):
    """Bar chart: total per-episode value, one bar per algo."""
    ep_totals = df.groupby(["algorithm", "episode"])[metric].sum().reset_index()
    summary   = ep_totals.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["mean"], yerr=summary["std"],
                  color=[COLORS.get(a, "gray") for a in summary["algorithm"]],
                  capsize=6, width=0.4, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["algorithm"], fontsize=11)
    _style_ax(ax, title, "Algorithm", ylabel)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved plot → {out}")


def plot_comparison_dashboard(df: pd.DataFrame):
    """4-panel dashboard saved as one PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("PPO vs SAC — P2P Energy Trading Comparison", fontsize=15, fontweight="bold")

    panels = [
        ("reward",           "Episode Reward per Step",          "Reward",       axes[0, 0]),
        ("p2p_volume_kwh",   "P2P Trading Volume (kWh/step)",    "kWh",          axes[0, 1]),
        ("total_import_kw",  "Grid Import per Step (kW)",        "kW",           axes[1, 0]),
        ("market_price",     "Market Clearing Price ($/kWh)",    "$/kWh",        axes[1, 1]),
    ]
    for metric, title, ylabel, ax in panels:
        for algo, grp in df.groupby("algorithm"):
            pivot = grp.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(pivot["step"], pivot["mean"],
                    label=algo, color=COLORS.get(algo, "gray"),
                    linestyle=STYLE.get(algo, "-"), linewidth=1.6)
            ax.fill_between(pivot["step"],
                            pivot["mean"] - pivot["std"],
                            pivot["mean"] + pivot["std"],
                            alpha=0.12, color=COLORS.get(algo, "gray"))
        _style_ax(ax, title, "Time Step", ylabel)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dashboard → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
_KPI_COLS = [
    "reward", "p2p_volume_kwh", "total_import_kw",
    "line_overload_kw", "failed_trades", "absolute_profit_usd",
]

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo, grp in df.groupby("algorithm"):
        ep_totals = grp.groupby("episode")[_KPI_COLS].sum()
        row = {"Algorithm": algo}
        for col in _KPI_COLS:
            row[f"{col}_mean"] = ep_totals[col].mean()
            row[f"{col}_std"]  = ep_totals[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PPO vs SAC — Comparative Summary (mean ± std over episodes)")
    print("=" * 70)
    kpi_labels = {
        "reward":             "Total Reward",
        "p2p_volume_kwh":     "P2P Volume (kWh)",
        "total_import_kw":    "Grid Import (kW)",
        "line_overload_kw":   "Line Overload (kW)",
        "failed_trades":      "Failed Trades",
        "absolute_profit_usd":"Profit (USD)",
    }
    for col, label in kpi_labels.items():
        print(f"\n  {label}:")
        for _, row in summary.iterrows():
            print(f"    {row['Algorithm']:>4s}: {row[f'{col}_mean']:>9.3f} ± {row[f'{col}_std']:.3f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs SAC on P2P Energy Trading")
    parser.add_argument("--ppo_model", type=str,
                        default="models_slim/seed_0/best_model.zip",
                        help="Path to the PPO best model .zip")
    parser.add_argument("--sac_model", type=str,
                        default="models_sac/best/best_model.zip",
                        help="Path to the SAC best model .zip")
    parser.add_argument("--data_file", type=str,
                        default="processed_hybrid_data.csv",
                        help="Evaluation dataset CSV (same for both)")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Evaluation episodes per algorithm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"\n[COMPARE] Loading PPO from: {args.ppo_model}")
    if not os.path.exists(args.ppo_model):
        raise FileNotFoundError(
            f"PPO model not found at '{args.ppo_model}'.\n"
            "Run PPO training first, or provide --ppo_model path."
        )
    ppo_model = PPO.load(args.ppo_model)

    print(f"[COMPARE] Loading SAC from: {args.sac_model}")
    if not os.path.exists(args.sac_model):
        raise FileNotFoundError(
            f"SAC model not found at '{args.sac_model}'.\n"
            "Run 'python train_sac/train_sac_hybrid.py' first, or provide --sac_model path."
        )
    sac_model = SAC.load(args.sac_model)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n[COMPARE] Evaluating PPO ({args.n_episodes} episodes)...")
    df_ppo = run_evaluation("PPO", ppo_model, args.data_file, args.n_episodes, args.seed)

    print(f"\n[COMPARE] Evaluating SAC ({args.n_episodes} episodes)...")
    df_sac = run_evaluation("SAC", sac_model, args.data_file, args.n_episodes, args.seed)

    df_all = pd.concat([df_ppo, df_sac], ignore_index=True)

    # Save raw metrics
    raw_path = os.path.join(RESULTS_DIR, "comparison_metrics.csv")
    df_all.to_csv(raw_path, index=False)
    print(f"\n[COMPARE] Raw metrics saved → {raw_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = build_summary(df_all)
    sum_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    summary.to_csv(sum_path, index=False)
    print_summary_table(summary)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[COMPARE] Generating plots...")
    plot_comparison_dashboard(df_all)
    plot_metric_over_steps(df_all, "reward",         "Reward per Step",           "Reward",   "plot_reward.png")
    plot_metric_over_steps(df_all, "p2p_volume_kwh", "P2P Trade Volume per Step", "kWh",      "plot_p2p_volume.png")
    plot_metric_over_steps(df_all, "total_import_kw","Grid Import per Step",      "kW",       "plot_grid_import.png")
    plot_episode_totals(df_all,    "p2p_volume_kwh", "Total P2P Volume / Episode","kWh",      "bar_p2p_total.png")
    plot_episode_totals(df_all,    "absolute_profit_usd","Total Profit / Episode","USD",      "bar_profit_total.png")

    print("\n[COMPARE] Done. Check train_sac/results/ for all outputs.")


if __name__ == "__main__":
    main()
