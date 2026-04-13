"""
Phase 14: From-Scratch Ablation Retraining
==========================================
Trains three SLIM v4 configurations **from scratch** at N=8 across multiple seeds
to prove **training-time causal necessity** of coordination incentives.

Modes:
  full_system   — use_alignment_reward=True,  use_curriculum=True
  no_alignment  — use_alignment_reward=False, use_curriculum=True
  no_curriculum — use_alignment_reward=True,  use_curriculum=False

Seeds: [42, 123, 999]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv

# ─── Config ────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = {
    "full_system":   {"use_alignment_reward": True,  "use_curriculum": True},
    "no_alignment":  {"use_alignment_reward": False, "use_curriculum": True},
    "no_curriculum": {"use_alignment_reward": True,  "use_curriculum": False},
}

N_AGENTS       = 8
SEEDS          = [42, 123, 999]
TRAIN_STEPS    = 200_000
CHECK_FREQ     = 2048
MODEL_BASE_DIR = "models_ablation_v6"
RESULTS_DIR    = "research_q1/results/ablation_v6"

os.makedirs(MODEL_BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ─── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {"full_system": "#2ECC71", "no_alignment": "#E74C3C", "no_curriculum": "#F39C12"}


# ─── Callback ─────────────────────────────────────────────────────────────────
class AblationCallback(BaseCallback):
    """Logs key metrics every check_freq steps."""

    def __init__(self, check_freq=CHECK_FREQ, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        # Sync curriculum step
        rel = self.n_calls * self.training_env.num_envs
        self.training_env.env_method("update_training_step", rel)

        if self.n_calls % self.check_freq == 0:
            info = self.locals["infos"][0]
            self.history.append({
                "step":          self.num_timesteps,
                "success":       info.get("trade_success_rate", 0.0),
                "grid_dep":      info.get("grid_dependency",    1.0),
                "p2p_volume":    info.get("p2p_volume",         0.0),
                "clean_profit":  info.get("clean_profit_usd",   0.0),
                "economic_profit": info.get("economic_profit_usd", 0.0),
                "beta":          info.get("curriculum_beta",    2.0),
            })
        return True


# ─── Single-run training ──────────────────────────────────────────────────────
def train_one(config_name: str, flags: dict, seed: int) -> list[dict]:
    print(f"\n  [{config_name} | seed={seed}] training {TRAIN_STEPS:,} steps …")

    train_env = VectorizedMultiAgentEnv(
        n_agents=N_AGENTS,
        use_alignment_reward=flags["use_alignment_reward"],
        use_curriculum=flags["use_curriculum"],
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", train_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
    )

    cb = AblationCallback()
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)

    save_dir = os.path.join(MODEL_BASE_DIR, f"{config_name}_s{seed}")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "model"))
    train_env.close()

    # Tag history with config / seed
    for row in cb.history:
        row["config"] = config_name
        row["seed"]   = seed
    return cb.history


# ─── Plotting ──────────────────────────────────────────────────────────────────
def plot_convergence(all_history: list[dict]):
    """6-panel convergence comparison with mean ± std shading across seeds."""
    df = pd.DataFrame(all_history)

    metrics = [
        ("success",         "Trade Success Rate",     "%"),
        ("grid_dep",        "Grid Dependency",        ""),
        ("p2p_volume",      "P2P Volume",             "kWh/step"),
        ("economic_profit", "Economic Profit (step)", "USD"),
        ("beta",            "Grid Penalty β",         ""),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    fig.suptitle("From-Scratch Ablation Convergence — N=8 (mean ± std, 3 seeds)",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, title, unit) in zip(axes, metrics):
        for cfg, color in PALETTE.items():
            sub = df[df["config"] == cfg]
            grouped = sub.groupby("step")[col]
            steps = grouped.mean().index.tolist()
            mean  = grouped.mean().values
            std   = grouped.std(ddof=0).fillna(0).values

            scale = 100 if "%" in unit else 1
            ax.plot(steps, mean * scale, label=cfg.replace("_", " ").title(),
                    color=color, linewidth=2.2)
            ax.fill_between(steps, (mean - std) * scale,
                            (mean + std) * scale, alpha=0.15, color=color)

        ax.set_title(f"{title}\n({unit})" if unit else title, fontsize=10)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "convergence_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def plot_final_bar(summary_df: pd.DataFrame):
    """Bar chart of final-checkpoint metrics per ablation config."""
    configs  = summary_df["config"].tolist()
    colors   = [PALETTE[c] for c in configs]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Final Performance: From-Scratch Ablation (N=8, 3-seed avg)",
                 fontsize=13, fontweight="bold")

    for ax, (col, label, scale) in zip(axes, [
        ("success",         "Success Rate (%)",     100),
        ("grid_dep",        "Grid Dependency (%)",  100),
        ("p2p_volume",      "P2P Volume (kWh)",       1),
        ("economic_profit", "Economic Profit ($)",    1),
    ]):
        vals = (summary_df[col] * scale).tolist()
        bars = ax.bar(configs, vals, color=colors, edgecolor="white", width=0.55)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "final_bar_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(config_name, flags, seed, eval_steps=1000):
    model_path = os.path.join(MODEL_BASE_DIR, f"{config_name}_s{seed}", "model")
    if not os.path.exists(model_path + ".zip"):
        return None

    model = PPO.load(model_path)
    eval_env = VectorizedMultiAgentEnv(
        n_agents=N_AGENTS,
        use_alignment_reward=flags["use_alignment_reward"],
        use_curriculum=flags["use_curriculum"],
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.env_method("update_training_step", 1_000_000)

    # Short warmup
    obs = eval_env.reset()
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = eval_env.step(action)

    obs = eval_env.reset()
    records = {"success": [], "grid_dep": [], "p2p_volume": [],
               "clean_profit": [], "economic_profit": []}

    for _ in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = eval_env.step(action)
        i = infos[0]
        records["success"].append(i.get("success", 0))
        records["grid_dep"].append(min(1.0, i.get("grid_dependency", 1.0)))
        records["p2p_volume"].append(i.get("p2p_volume", 0))
        records["clean_profit"].append(i.get("clean_profit_usd", 0))
        records["economic_profit"].append(i.get("economic_profit_usd", 0))

    eval_env.close()
    return {k: float(np.mean(v)) for k, v in records.items()}


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHASE 14: From-Scratch Ablation Retraining  (N=8, 3 seeds)")
    print("=" * 60)

    all_history = []

    # ── Training ──────────────────────────────────────────────────────────────
    for config_name, flags in ABLATION_CONFIGS.items():
        print(f"\n{'─'*55}\nTraining: {config_name.upper()}")
        for seed in SEEDS:
            history = train_one(config_name, flags, seed)
            all_history.extend(history)

    # Save raw history
    hist_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nHistory saved → {hist_path}")

    # ── Convergence plots ─────────────────────────────────────────────────────
    print("\nGenerating convergence plots …")
    plot_convergence(all_history)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating trained models …")
    eval_rows = []
    for config_name, flags in ABLATION_CONFIGS.items():
        seed_results = []
        for seed in SEEDS:
            res = evaluate_model(config_name, flags, seed)
            if res:
                seed_results.append(res)
        if seed_results:
            avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
            avg["config"] = config_name
            eval_rows.append(avg)
            print(f"  {config_name:15s}  success={avg['success']*100:.1f}%  "
                  f"grid={avg['grid_dep']*100:.1f}%  "
                  f"p2p={avg['p2p_volume']:.4f} kWh  "
                  f"econ_profit=${avg['economic_profit']:.2f}")

    summary_df = pd.DataFrame(eval_rows)
    csv_path = os.path.join(RESULTS_DIR, "ablation_v6_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved → {csv_path}")

    plot_final_bar(summary_df)

    print("\n✅  Phase 14 ablation retraining complete.")
    print(f"    Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
