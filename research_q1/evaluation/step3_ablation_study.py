"""
step3_ablation_study.py
========================
Ablation study validating design choices in the SLIM v2 reward function.

TWO-TIER approach (per task specification):
  A) POST-HOC ablation  — same trained model, reward component zeroed at eval-time
  B) RETRAINED ablation — new PPO model trained 100k steps with component REMOVED

Ablation configurations:
  1. Full system          (original, baseline)
  2. No CO2 penalty       [post-hoc + retrained]
  3. No fairness penalty  [post-hoc + retrained]
  4. No Lagrangian        [post-hoc only]
  5. Early checkpoint     [no curriculum, uses 100k-step checkpoint]

Metrics per config:
  - Cumulative clean profit (USD)
  - Cumulative CO2 (kg)
  - Total P2P volume (kWh)
  - Mean grid import (kW)
  - Active buyers fraction (agents with |trade_intent| > 0.01)

Key finding reported:
  Nash equilibrium collapse — P2P volume near zero regardless of ablation config.
  Identifies which component drives collapse.

Outputs:
  research_q1/results/step3_ablation_results.csv
  research_q1/results/step3_ablation_chart.png
  research_q1/results/models/model_no_fairness.zip
  research_q1/results/models/model_no_co2.zip
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Import base env at module level so subclasses can find it
from train.energy_env_robust import EnergyMarketEnvRobust as EnergyMarketEnvRobustBase

OUT_DIR        = os.path.join(ROOT, "research_q1", "results")
MODELS_OUT_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUT_DIR, exist_ok=True)

N_AGENTS       = 4
TIMESTEP_HOURS = 1.0
RETAIL_OFF_PEAK = 0.20
RETAIL_PEAK     = 0.50
PEAK_HOURS      = set(range(17, 21))
CARBON_INTENSITY = 0.233
MODEL_PATH      = os.path.join(ROOT, "models_slim", "seed_0", "best_model.zip")
CHECKPOINT_PATH = os.path.join(ROOT, "models_slim", "seed_0", "ppo_slim_100000_steps.zip")
DATA_PATH       = os.path.join(ROOT, "processed_hybrid_data.csv")
N_EPISODES_EVAL = 3
TRAIN_TIMESTEPS = 100_000
SEED            = 42

ABLATION_MODEL_PATHS = {
    "no_fairness_retrained": os.path.join(MODELS_OUT_DIR, "model_no_fairness.zip"),
    "no_co2_retrained":      os.path.join(MODELS_OUT_DIR, "model_no_co2.zip"),
}


def retail_price(step: int) -> float:
    return RETAIL_PEAK if (step % 24) in PEAK_HOURS else RETAIL_OFF_PEAK


# ── ENV VARIANTS FOR RETRAINING ───────────────────────────────────────────────

class NoFairnessEnv(EnergyMarketEnvRobustBase):
    """
    Proper gymnasium subclass: fairness_coeff = 0 throughout training.
    Registered as a real gym.Env so DummyVecEnv accepts it.
    """
    def __init__(self, data_path: str):
        super().__init__(
            n_agents=N_AGENTS, data_file=data_path, random_start_day=True,
            enable_ramp_rates=True, enable_losses=True, forecast_horizon=4,
            enable_predictive_obs=True, forecast_noise_std=0.05, diversity_mode=True,
        )
        # Zero out fairness after parent __init__ sets it up
        self.reward_tracker.fairness_coeff = 0.0


class NoCO2Env(EnergyMarketEnvRobustBase):
    """
    Proper gymnasium subclass: CO2 penalty coefficient = 0 throughout training.
    Overrides _step_grid_and_reward to add back the CO2 penalty refund.
    """
    def __init__(self, data_path: str):
        super().__init__(
            n_agents=N_AGENTS, data_file=data_path, random_start_day=True,
            enable_ramp_rates=True, enable_losses=True, forecast_horizon=4,
            enable_predictive_obs=True, forecast_noise_std=0.05, diversity_mode=True,
        )

    def _step_grid_and_reward(self, physics_state, market_results):
        obs, reward, info = super()._step_grid_and_reward(physics_state, market_results)
        # Refund the CO2 penalty that was subtracted from reward
        refund = float(self.reward_tracker.step_co2_penalty.sum())
        reward += refund
        self.reward_tracker.step_co2_penalty[:] = 0.0
        return obs, reward, info



# ── RETRAINING ────────────────────────────────────────────────────────────────

def retrain_ablation_model(env_class, data_path: str,
                           save_path: str, label: str) -> str:
    """Train PPO on ablation env for TRAIN_TIMESTEPS steps."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if os.path.exists(save_path):
        print(f"  [SKIP] {label} model already exists: {save_path}")
        return save_path

    print(f"\n  [RETRAIN] {label} -- {TRAIN_TIMESTEPS} steps ...")
    t0 = time.time()

    def make_env():
        return env_class(data_path=data_path)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    model.save(save_path)

    elapsed = time.time() - t0
    print(f"  [RETRAIN] Done in {elapsed:.0f}s -> {save_path}")
    return save_path


# ── EVALUATION WITH ACTIVE BUYER DETECTION ────────────────────────────────────

def make_eval_env(data_path: str, co2_off: bool = False, fairness_off: bool = False,
                  lag_off: bool = False):
    """Create a base env with optional post-hoc ablations applied."""
    import types
    from train.energy_env_robust import EnergyMarketEnvRobust

    env = EnergyMarketEnvRobust(
        n_agents=N_AGENTS, data_file=data_path, random_start_day=False,
        enable_ramp_rates=True, enable_losses=True, forecast_horizon=4,
        enable_predictive_obs=True, forecast_noise_std=0.0, diversity_mode=True,
    )

    if fairness_off:
        env.reward_tracker.fairness_coeff = 0.0

    if co2_off or lag_off:
        original_fn = env._step_grid_and_reward.__func__

        def patched(inner_self, physics_state, market_results):
            obs, reward, info = original_fn(inner_self, physics_state, market_results)
            if co2_off:
                refund = float(inner_self.reward_tracker.step_co2_penalty.sum())
                reward += refund
                inner_self.reward_tracker.step_co2_penalty[:] = 0.0
            if lag_off:
                lag_pen = info.get("lagrangian/penalty_this_step", 0.0)
                reward += float(lag_pen)
                info["lagrangian/penalty_this_step"] = 0.0
            return obs, reward, info

        env._step_grid_and_reward = types.MethodType(patched, env)

    return env


TRADE_INTENT_THRESHOLD = 0.01  # kW — below this is considered "no intent"


def run_config_eval(model, config_label: str, ablation_type: str,
                    n_episodes: int, seed: int, data_path: str,
                    co2_off=False, fairness_off=False, lag_off=False) -> list[dict]:
    """Evaluate one ablation config, return list of episode-level summary dicts."""
    from stable_baselines3 import PPO

    episode_rows = []
    for ep in range(n_episodes):
        env = make_eval_env(data_path, co2_off=co2_off,
                            fairness_off=fairness_off, lag_off=lag_off)
        obs, _ = env.reset(seed=seed + ep)
        done, step = False, 0
        ep_profit = ep_co2 = ep_p2p = ep_grid = 0.0
        active_buyer_count = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Track raw action to diagnose trade intent
            raw_action = action.reshape(N_AGENTS, 3)
            trade_intents = raw_action[:, 1]

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            r_price  = retail_price(step)
            tr       = info.get("mean_profit", 0.0) * N_AGENTS
            gc       = info.get("total_import", 0.0) * TIMESTEP_HOURS * r_price
            ep_profit += tr - gc
            ep_co2    += info.get("total_import", 0.0) * TIMESTEP_HOURS * CARBON_INTENSITY
            ep_p2p    += info.get("p2p_volume_kwh_step", 0.0)
            ep_grid   += info.get("total_import", 0.0)

            buyers = np.sum(np.abs(trade_intents) > TRADE_INTENT_THRESHOLD)
            active_buyer_count += int(buyers)
            step_count += 1
            step += 1

        episode_rows.append({
            "config":                config_label,
            "ablation_type":         ablation_type,
            "episode":               ep,
            "cum_clean_profit_usd":  ep_profit,
            "cum_co2_kg":            ep_co2,
            "total_p2p_kwh":         ep_p2p,
            "mean_grid_import_kw":   ep_grid / max(step_count, 1),
            "active_buyers_per_step": active_buyer_count / max(step_count, 1),
        })

    return episode_rows


def print_ablation_table(df: pd.DataFrame):
    summary = df.groupby(["config", "ablation_type"])[[
        "cum_clean_profit_usd", "cum_co2_kg",
        "total_p2p_kwh", "mean_grid_import_kw", "active_buyers_per_step"
    ]].agg(["mean", "std"]).reset_index()

    print("\n" + "="*75)
    print("  STEP 3: Ablation Study Results")
    print("="*75)
    f = "{:<30} {:<12} {:>12.3f} {:>12.3f} {:>12.4f}"
    print(f"  {'Config':<30} {'Type':<12} {'Profit(USD)':>12} {'CO2(kg)':>12} {'P2P(kWh)':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for _, row in df.groupby(["config", "ablation_type"]).mean().reset_index().iterrows():
        print(f"  {row['config']:<30} {row['ablation_type']:<12}"
              f" {row['cum_clean_profit_usd']:>12.3f}"
              f" {row['cum_co2_kg']:>12.3f}"
              f" {row['total_p2p_kwh']:>12.4f}")
    print("="*75)

    if df["total_p2p_kwh"].max() < 1.0:
        print("\n  [FIND] All configs show near-zero P2P volume.")
        print("         Nash equilibrium collapse is ROBUST to reward shaping changes.")
        print("         The collapse is driven by policy structure, not reward components.")


def make_ablation_chart(df: pd.DataFrame, out_dir: str):
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("ggplot")

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    summary = df.groupby(["config", "ablation_type"]).agg(
        profit_mean=("cum_clean_profit_usd", "mean"),
        profit_std=("cum_clean_profit_usd", "std"),
        co2_mean=("cum_co2_kg", "mean"),
        co2_std=("cum_co2_kg", "std"),
        p2p_mean=("total_p2p_kwh", "mean"),
        grid_mean=("mean_grid_import_kw", "mean"),
    ).reset_index()

    configs = summary["config"].tolist()
    types   = summary["ablation_type"].tolist()
    n       = len(summary)
    x       = np.arange(n)
    bar_colors = [colors[0] if t == "posthoc" else colors[1] for t in types]

    metrics = [
        ("profit_mean", "profit_std", "Cumulative Clean Profit (USD)", "profit"),
        ("co2_mean",    "co2_std",    "Cumulative CO2 (kg)",          "co2"),
        ("p2p_mean",    None,         "Total P2P Volume (kWh)",       "p2p"),
        ("grid_mean",   None,         "Mean Grid Import (kW)",        "grid"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Step 3: Ablation Study — Component Contribution Analysis",
                 fontsize=13, fontweight="bold")

    short_labels = [c.split("_")[0] + "\n" + t[:4] for c, t in zip(configs, types)]

    for (mean_col, std_col, ylabel, key), ax in zip(metrics, axes.flat):
        vals  = summary[mean_col].values
        errs  = summary[std_col].values if std_col else None
        bars  = ax.bar(x, vals, yerr=errs, color=bar_colors, capsize=4,
                       alpha=0.8, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="grey", lw=0.5, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend for ablation type
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=colors[0], label="Post-hoc"),
                    Patch(facecolor=colors[1], label="Retrained")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = os.path.join(out_dir, "step3_ablation_chart.png")
    plt.savefig(fname)
    plt.savefig(fname.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved -> {fname}")


def main():
    from stable_baselines3 import PPO

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found:", MODEL_PATH)
        return

    print("[STEP 3] Loading base model:", MODEL_PATH)
    base_model = PPO.load(MODEL_PATH)

    # ── PHASE A: LIGHTWEIGHT RETRAINING ───────────────────────────────────────
    retrain_ablation_model(
        NoFairnessEnv, DATA_PATH,
        ABLATION_MODEL_PATHS["no_fairness_retrained"],
        "No-Fairness"
    )
    retrain_ablation_model(
        NoCO2Env, DATA_PATH,
        ABLATION_MODEL_PATHS["no_co2_retrained"],
        "No-CO2"
    )

    # Load retrained models
    model_no_fair = PPO.load(ABLATION_MODEL_PATHS["no_fairness_retrained"])
    model_no_co2  = PPO.load(ABLATION_MODEL_PATHS["no_co2_retrained"])

    # Load early checkpoint (no curriculum = trained without later phases)
    if os.path.exists(CHECKPOINT_PATH):
        model_early = PPO.load(CHECKPOINT_PATH)
    else:
        print(f"  [WARN] Early checkpoint not found at {CHECKPOINT_PATH}. Skipping.")
        model_early = None

    # ── PHASE B: EVALUATION ───────────────────────────────────────────────────
    all_rows = []

    configs_posthoc = [
        ("full_system",      False, False, False, base_model),
        ("no_co2_pen",       True,  False, False, base_model),
        ("no_fairness_pen",  False, True,  False, base_model),
        ("no_lagrangian",    False, False, True,  base_model),
    ]

    print("\n  [EVAL] Post-hoc ablations ...")
    for (label, co2_off, fair_off, lag_off, mdl) in configs_posthoc:
        rows = run_config_eval(mdl, label, "posthoc",
                               N_EPISODES_EVAL, SEED, DATA_PATH,
                               co2_off=co2_off, fairness_off=fair_off, lag_off=lag_off)
        all_rows.extend(rows)
        mean_profit = np.mean([r["cum_clean_profit_usd"] for r in rows])
        mean_p2p    = np.mean([r["total_p2p_kwh"] for r in rows])
        print(f"    {label:<24} profit={mean_profit:+.2f}  p2p={mean_p2p:.4f}")

    print("\n  [EVAL] Retrained ablation models ...")
    rows_nf = run_config_eval(model_no_fair, "no_fairness_pen", "retrained",
                              N_EPISODES_EVAL, SEED, DATA_PATH)
    all_rows.extend(rows_nf)
    mean_p = np.mean([r["cum_clean_profit_usd"] for r in rows_nf])
    mean_p2p = np.mean([r["total_p2p_kwh"] for r in rows_nf])
    print(f"    {'no_fairness_pen (retrained)':<24} profit={mean_p:+.2f}  p2p={mean_p2p:.4f}")

    rows_nc = run_config_eval(model_no_co2, "no_co2_pen", "retrained",
                              N_EPISODES_EVAL, SEED, DATA_PATH,
                              co2_off=True)
    all_rows.extend(rows_nc)
    mean_p = np.mean([r["cum_clean_profit_usd"] for r in rows_nc])
    mean_p2p = np.mean([r["total_p2p_kwh"] for r in rows_nc])
    print(f"    {'no_co2_pen (retrained)':<24} profit={mean_p:+.2f}  p2p={mean_p2p:.4f}")

    if model_early is not None:
        print("\n  [EVAL] Early checkpoint (no curriculum) ...")
        rows_early = run_config_eval(model_early, "no_curriculum_early", "posthoc",
                                     N_EPISODES_EVAL, SEED, DATA_PATH)
        all_rows.extend(rows_early)
        mean_p = np.mean([r["cum_clean_profit_usd"] for r in rows_early])
        mean_p2p = np.mean([r["total_p2p_kwh"] for r in rows_early])
        print(f"    {'no_curriculum_early':<24} profit={mean_p:+.2f}  p2p={mean_p2p:.4f}")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "step3_ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Ablation results saved -> {csv_path}")

    print_ablation_table(df)
    make_ablation_chart(df, OUT_DIR)

    print("[STEP 3] COMPLETE\n")


if __name__ == "__main__":
    main()
