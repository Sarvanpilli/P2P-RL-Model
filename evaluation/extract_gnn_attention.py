"""
evaluation/extract_gnn_attention.py

Extracts real GATv2Conv attention weights from the trained CTDEGNNPolicy
using the built-in extract_attention() method.

Usage:
    python evaluation/extract_gnn_attention.py

Outputs:
    - Prints the mean 4×4 attention weight matrix to stdout
    - Saves evaluation/results/attention_weights_matrix.csv
    - Saves research_q1/results/gnn_attention_heatmap.png (300 DPI)
"""

import os
import sys
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM
from train.energy_env_recovery import EnergyMarketEnvRecovery
from research_q1.novelty.gnn_policy import CTDEGNNPolicy

# ── config ────────────────────────────────────────────────────────────────────
N_AGENTS = 4
CHECKPOINT = "research_q1/models/slim_ppo/slim_ppo_250000_steps.zip"
EVAL_STEPS = 1_752       # 20% of 8760-hour dataset (eval split)
OUTPUT_DIR = "evaluation/results"
HEATMAP_OUT = "research_q1/results/gnn_attention_heatmap.png"
CSV_OUT = os.path.join(OUTPUT_DIR, "attention_weights_matrix.csv")
AGENT_NAMES = ["Solar", "Wind", "EV/V2G", "Standard"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HEATMAP_OUT), exist_ok=True)


def build_mean_attention_matrix(model, env, n_steps: int):
    """
    Run n_steps evaluation steps.
    On each step, call policy.extract_attention(obs) to get real GATv2 weights.

    Returns
    -------
    mean_matrix : np.ndarray, shape (N_AGENTS, N_AGENTS)
        Mean attention weight from agent[row] → agent[col].
    hourly_weights : list of list  shape (24, n_edges)
        Mean per-hour attention weights for the 24-hour time series.
    """
    policy = model.policy
    obs_tensor_fn = lambda obs: torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    weight_sum = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float64)
    count = 0

    # For time-series plot: bucket attention sums by hour-of-day
    hourly_sum = np.zeros((24, N_AGENTS, N_AGENTS))
    hourly_count = np.zeros(24, dtype=int)

    obs, _ = env.reset(seed=42)
    hour = 0

    for step in range(n_steps):
        obs_t = obs_tensor_fn(obs)

        with torch.no_grad():
            edges, alpha = policy.extract_attention(obs_t)
            # alpha: [B*n_edges, n_heads]  →  average over heads → [n_edges,]
            if alpha.ndim > 1:
                alpha = alpha.mean(dim=-1)
            alpha_np = alpha.detach().cpu().numpy()  # [n_edges,]

            # edges: [2, B*n_edges]
            src = edges[0].detach().cpu().numpy() % N_AGENTS
            dst = edges[1].detach().cpu().numpy() % N_AGENTS

        for s, d, w in zip(src, dst, alpha_np):
            weight_sum[int(s), int(d)] += float(w)
            hourly_sum[hour % 24, int(s), int(d)] += float(w)

        count += 1
        hourly_count[hour % 24] += 1

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        hour += 1
        if terminated or truncated:
            obs, _ = env.reset()

    mean_matrix = weight_sum / max(count, 1)
    # Per-hour matrix (avoid divide-by-zero)
    denom = np.maximum(hourly_count, 1)
    hourly_mean = hourly_sum / denom[:, np.newaxis, np.newaxis]  # (24, N, N)

    return mean_matrix, hourly_mean


def plot_heatmap_and_timeseries(mean_matrix, hourly_mean, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Subplot 1: 4×4 Attention Heatmap ─────────────────────────────────────
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        xticklabels=AGENT_NAMES,
        yticklabels=AGENT_NAMES,
        ax=axes[0],
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_xlabel("Target Agent (being attended to)", fontsize=11)
    axes[0].set_ylabel("Source Agent", fontsize=11)
    axes[0].set_title("GATv2 Attention Weights\nMean over Evaluation Episode", fontsize=12, fontweight="bold")

    # ── Subplot 2: 24-hour time series for Agent 2 → Agent 0 ─────────────────
    hours = np.arange(24)
    ev_to_solar = hourly_mean[:, 2, 0]   # EV/V2G → Solar
    ev_to_wind  = hourly_mean[:, 2, 1]   # EV/V2G → Wind

    axes[1].plot(hours, ev_to_solar, marker="o", linewidth=2, label="EV/V2G → Solar", color="#e05c2e")
    axes[1].plot(hours, ev_to_wind,  marker="s", linewidth=2, label="EV/V2G → Wind",  color="#2980b9", linestyle="--")

    # Solar peak band
    axes[1].axvspan(10, 15, alpha=0.1, color="gold", label="Solar peak (10–15h)")
    # Evening peak band
    axes[1].axvspan(17, 21, alpha=0.1, color="red", label="Grid peak (17–21h)")

    axes[1].set_xlabel("Hour of Day", fontsize=11)
    axes[1].set_ylabel("Mean Attention Weight", fontsize=11)
    axes[1].set_title("Attention Weight vs. Hour of Day\n(Agent 2/EV/V2G → Agent 0/Solar)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def main():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        print("Please update CHECKPOINT at the top of this script.")
        sys.exit(1)

    print(f"Loading model: {CHECKPOINT}")
    # Force EnergyMarketEnvRecovery with forecast_horizon=4 to match 88-dim observation space
    # 7 base + 2 weather + 4 type + 8 (2*4) forecast + 1 peer = 22 features * 4 agents = 88
    env = EnergyMarketEnvRecovery(n_agents=N_AGENTS, forecast_horizon=4)
    model = PPO.load(CHECKPOINT, env=env)

    # Ensure the policy is CTDEGNNPolicy
    if not hasattr(model.policy, "extract_attention"):
        print("ERROR: The loaded model does not have `extract_attention()`.")
        print("This model was likely trained with a standard MlpPolicy, not CTDEGNNPolicy.")
        print("Run evaluation/extract_attention_weights.py instead (fallback script).")
        sys.exit(1)

    print(f"Policy type: {type(model.policy).__name__} ✓")
    print(f"Running {EVAL_STEPS} eval steps to extract attention ...\n")

    mean_matrix, hourly_mean = build_mean_attention_matrix(model, env, EVAL_STEPS)
    env.close()

    # ── Print matrix ──────────────────────────────────────────────────────────
    print("Mean GATv2Conv Attention Weight Matrix (row=source, col=target):\n")
    header = f"{'':12}" + "".join(f"{n:>12}" for n in AGENT_NAMES)
    print(header)
    for i, name in enumerate(AGENT_NAMES):
        row = "".join(f"{mean_matrix[i, j]:>12.4f}" for j in range(N_AGENTS))
        print(f"{name:<12}{row}")

    max_idx = np.unravel_index(np.argmax(mean_matrix), mean_matrix.shape)
    print(f"\nHighest attention: {AGENT_NAMES[max_idx[0]]} → {AGENT_NAMES[max_idx[1]]}  "
          f"(weight = {mean_matrix[max_idx]:.4f})")

    # Per-hour peak
    ev_to_solar_peak_hour = int(np.argmax(hourly_mean[:, 2, 0]))
    print(f"EV/V2G → Solar attention peaks at hour {ev_to_solar_peak_hour:02d}:00")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From \\ To"] + AGENT_NAMES)
        for i, name in enumerate(AGENT_NAMES):
            writer.writerow([name] + [f"{mean_matrix[i, j]:.4f}" for j in range(N_AGENTS)])
    print(f"Matrix saved to: {CSV_OUT}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_heatmap_and_timeseries(mean_matrix, hourly_mean, HEATMAP_OUT)


if __name__ == "__main__":
    main()
