"""
Phase 13: Publication-quality plots for SLIM v4 Final Scientific Validation.
Generates enhanced comparison charts from the ablation and scalability results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ──────────────────────────────────────────────────────────
RESULTS_DIR = "research_q1/results/final_validation"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "publication_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Colour palette ───────────────────────────────────────
PALETTE = {
    "Full System":   "#2ECC71",   # green
    "No Alignment":  "#E74C3C",   # red
    "No Curriculum": "#F39C12",   # amber
    "grid":          "#E74C3C",
    "p2p":           "#2ECC71",
    "neutral":       "#3498DB",
}
FONT = dict(family="DejaVu Sans")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

# ─── Data ─────────────────────────────────────────────────
ablation_df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
scalability_df = pd.read_csv(os.path.join(RESULTS_DIR, "scalability_validation.csv"))

# Parse agent counts for scalability
scalability_df["N"] = scalability_df["Config"].str.extract(r"(\d+)").astype(int)

# ══════════════════════════════════════════════════════════
# FIGURE 1 – Ablation Multi-Metric Summary (4-panel)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Ablation Study: Impact of Coordination Components (N=8 Agents)",
             fontsize=15, fontweight="bold", y=1.02)

configs = ablation_df["Config"].tolist()
colors  = [PALETTE.get(c, "#3498DB") for c in configs]

metrics = [
    ("Success %",     "Trade Success Rate (%)",     True),
    ("Grid %",        "Grid Dependency (%)",         False),
    ("P2P Vol",       "Mean P2P Volume (kWh/step)",  True),
    ("Economic ($)",  "Cumulative Economic Profit ($)", True),
]

for ax, (col, label, higher_better) in zip(axes, metrics):
    vals = ablation_df[col].tolist()
    bars = ax.bar(configs, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    ax.set_title(label, fontsize=11, pad=8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(bar.get_height())*0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    # Highlight best bar
    best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "fig1_ablation_4panel.png")
plt.savefig(path1, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")

# ══════════════════════════════════════════════════════════
# FIGURE 2 – Scalability: Success, Grid, P2P, Carbon
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
fig.suptitle("SLIM v4 Scalability: Performance Across Agent Densities",
             fontsize=15, fontweight="bold")

N = scalability_df["N"].tolist()

# --- Success rate
ax0.plot(N, scalability_df["Success %"], "o-", color=PALETTE["p2p"],
         linewidth=2.5, markersize=8, label="Success %")
ax0.fill_between(N, scalability_df["Success %"], alpha=0.15, color=PALETTE["p2p"])
ax0.set_title("Trade Success Rate", fontsize=12)
ax0.set_xlabel("# Agents"); ax0.set_ylabel("Success Rate (%)")
for x, y in zip(N, scalability_df["Success %"]):
    ax0.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=9)

# --- Grid dependency
ax1.plot(N, scalability_df["Grid %"], "s-", color=PALETTE["grid"],
         linewidth=2.5, markersize=8, label="Grid %")
ax1.fill_between(N, scalability_df["Grid %"], alpha=0.12, color=PALETTE["grid"])
ax1.set_title("Grid Dependency", fontsize=12)
ax1.set_xlabel("# Agents"); ax1.set_ylabel("Grid Dependency (%)")
for x, y in zip(N, scalability_df["Grid %"]):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=9)

# --- P2P Volume (log scale due to large range)
ax2.semilogy(N, scalability_df["P2P Vol"], "^-", color=PALETTE["neutral"],
             linewidth=2.5, markersize=8)
ax2.fill_between(N, scalability_df["P2P Vol"], alpha=0.12, color=PALETTE["neutral"])
ax2.set_title("P2P Market Volume (log scale)", fontsize=12)
ax2.set_xlabel("# Agents"); ax2.set_ylabel("Mean P2P Volume (kWh/step, log)")
for x, y in zip(N, scalability_df["P2P Vol"]):
    ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8)

# --- Carbon Emissions
ax3.bar(N, scalability_df["Carbon (kg)"] / 1000.0,
        color=["#AAB7B8", "#7F8C8D", "#566573", "#2C3E50"],
        width=1.5, edgecolor="white")
ax3.set_title("Cumulative Carbon Emissions", fontsize=12)
ax3.set_xlabel("# Agents"); ax3.set_ylabel("Carbon Emissions (tonnes CO₂)")
ax3.set_xticks(N)
for x, y in zip(N, scalability_df["Carbon (kg)"] / 1000.0):
    ax3.text(x, y + 0.3, f"{y:.1f}t", ha="center", fontsize=9)

path2 = os.path.join(PLOTS_DIR, "fig2_scalability_4panel.png")
plt.savefig(path2, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")

# ══════════════════════════════════════════════════════════
# FIGURE 3 – Economic Profit Waterfall (Ablation)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Economic Profit Comparison Across Ablation Modes",
             fontsize=14, fontweight="bold")

ep_vals   = ablation_df["Economic ($)"].tolist()
bar_colors = [PALETTE.get(c, "#3498DB") for c in configs]
bars = ax.barh(configs, ep_vals, color=bar_colors, edgecolor="white", height=0.5)

# Vertical zero line
ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
ax.set_xlabel("Cumulative Economic Profit (USD)", fontsize=11)
ax.set_ylabel("")

for bar, val in zip(bars, ep_vals):
    xpos = val - 5 if val < 0 else val + 2
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f"${val:.2f}",
            va="center", ha="right" if val < 0 else "left", fontsize=10)

ax.set_facecolor("#FAFAFA")
path3 = os.path.join(PLOTS_DIR, "fig3_economic_profit_ablation.png")
plt.savefig(path3, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {path3}")

# ══════════════════════════════════════════════════════════
# FIGURE 4 – Liquidity Scaling (annotated)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Market Liquidity Scales Super-Linearly with Agent Density",
             fontsize=13, fontweight="bold")

p2p = scalability_df["P2P Vol"].tolist()
ax.plot(N, p2p, "o-", color="#27AE60", linewidth=3, markersize=10, zorder=3)
ax.fill_between(N, p2p, alpha=0.15, color="#27AE60")

# Annotate growth multiples from N=4
base = p2p[0]
for xi, yi in zip(N, p2p):
    mult = yi / base
    ax.annotate(f"{mult:.1f}×\n({yi:.4f} kWh)",
                xy=(xi, yi), xytext=(xi, yi * 1.18),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))

ax.set_xlabel("Number of Agents", fontsize=12)
ax.set_ylabel("Mean P2P Volume per Step (kWh)", fontsize=12)
ax.set_xticks(N)
ax.set_xticklabels([f"N={n}" for n in N], fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)

path4 = os.path.join(PLOTS_DIR, "fig4_liquidity_scaling_annotated.png")
plt.savefig(path4, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {path4}")

print("\n✅  All publication plots saved to:", PLOTS_DIR)
