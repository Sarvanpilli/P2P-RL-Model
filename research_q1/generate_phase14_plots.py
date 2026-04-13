"""
Phase 14: Updated Publication Plots
Combines Phase 13 validation + Phase 14 from-scratch ablation results
into a unified, publication-quality visualisation suite.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Paths ─────────────────────────────────────────────────────────────────────
ABLATION_V6_DIR  = "research_q1/results/ablation_v6"
PHASE13_DIR      = "research_q1/results/final_validation"
OUT_DIR          = "research_q1/results/phase14_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PAL = {
    "full_system":   "#2ECC71",
    "no_alignment":  "#E74C3C",
    "no_curriculum": "#F39C12",
}
LABELS = {
    "full_system":   "Full System",
    "no_alignment":  "No Alignment",
    "no_curriculum": "No Curriculum",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A – Convergence curves (from-scratch, 3-seed mean ± std)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure A: Convergence curves …")

with open(os.path.join(ABLATION_V6_DIR, "training_history.json")) as f:
    history = json.load(f)

df_hist = pd.DataFrame(history)

metrics_conv = [
    ("success",   "Trade Success Rate (%)",    100, True),
    ("grid_dep",  "Grid Dependency (%)",        100, False),
    ("p2p_volume","P2P Volume (kWh/step)",        1, True),
    ("economic_profit", "Economic Profit ($/step)", 1, True),
    ("beta",      "Grid Penalty β",              1, False),
]

fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.suptitle(
    "Phase 14: From-Scratch Ablation — Convergence Curves (N=8, 3 seeds, mean ± std)",
    fontsize=14, fontweight="bold", y=1.03
)

for ax, (col, title, scale, _higher) in zip(axes, metrics_conv):
    for cfg, color in PAL.items():
        sub = df_hist[df_hist["config"] == cfg].groupby("step")[col]
        steps = sub.mean().index.tolist()
        mean  = sub.mean().values * scale
        std   = sub.std(ddof=0).fillna(0).values * scale
        ax.plot(steps, mean, label=LABELS[cfg], color=color, linewidth=2.2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Training Step", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].legend(fontsize=8, loc="lower right")
plt.tight_layout()
path_a = os.path.join(OUT_DIR, "figA_convergence_curves.png")
plt.savefig(path_a, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_a}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B – Final performance bars (from-scratch evaluation, 3-seed avg)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure B: Final performance bars …")

summary = pd.read_csv(os.path.join(ABLATION_V6_DIR, "ablation_v6_summary.csv"))
configs = ["full_system", "no_alignment", "no_curriculum"]
summary = summary.set_index("config").loc[configs].reset_index()

metrics_bar = [
    ("success",         "Trade Success Rate (%)", 100),
    ("grid_dep",        "Grid Dependency (%)",    100),
    ("p2p_volume",      "P2P Volume (kWh/step)",    1),
    ("economic_profit", "Economic Profit ($/step)", 1),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    "Phase 14: Final Evaluation — From-Scratch Ablation (3-seed average, N=8)",
    fontsize=13, fontweight="bold"
)

for ax, (col, label, scale) in zip(axes, metrics_bar):
    vals   = (summary[col] * scale).tolist()
    colors = [PAL[c] for c in summary["config"]]
    xlabs  = [LABELS[c] for c in summary["config"]]
    bars   = ax.bar(xlabs, vals, color=colors, edgecolor="white", width=0.55)
    ax.set_title(label, fontsize=11, pad=8)
    ax.set_xticklabels(xlabs, rotation=12, ha="right", fontsize=9)
    # Best marker
    best_idx = int(np.argmax(vals)) if col not in ("grid_dep",) else int(np.argmin(vals))
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(3)
    for bar, v in zip(bars, vals):
        yoff = abs(v) * 0.03
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (yoff if v >= 0 else -yoff * 3),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
path_b = os.path.join(OUT_DIR, "figB_final_bars.png")
plt.savefig(path_b, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_b}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C – Head-to-head: Phase 13 (inference toggle) vs Phase 14 (scratch)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure C: Phase 13 vs Phase 14 head-to-head …")

p13 = pd.read_csv(os.path.join(PHASE13_DIR, "ablation_results.csv"))
p13_map = {
    "Full System":   "full_system",
    "No Alignment":  "no_alignment",
    "No Curriculum": "no_curriculum",
}
p13["config"] = p13["Config"].map(p13_map)
p13 = p13.set_index("config")

p14 = summary.set_index("config")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Phase 13 (Inference Toggle) vs Phase 14 (From Scratch)\n"
    "Proving Training-Time Causal Importance of Coordination Incentives",
    fontsize=12, fontweight="bold"
)

compare_metrics = [
    ("Success %", "success",   100, "Trade Success Rate (%)"),
    ("Grid %",    "grid_dep",  100, "Grid Dependency (%)"),
    ("P2P Vol",   "p2p_volume",  1, "P2P Volume (kWh/step)"),
]

x = np.arange(3)
w = 0.35
cfg_order = ["full_system", "no_alignment", "no_curriculum"]

for ax, (p13_col, p14_col, scale, label) in zip(axes, compare_metrics):
    p13_vals = [p13.loc[c, p13_col] * (1 if scale == 1 else 1) for c in cfg_order]
    p14_vals = [p14.loc[c, p14_col] * scale for c in cfg_order]

    b1 = ax.bar(x - w/2, p13_vals, w, label="Ph13: Inference Toggle",
                color=[PAL[c] for c in cfg_order], alpha=0.6, edgecolor="white")
    b2 = ax.bar(x + w/2, p14_vals, w, label="Ph14: Trained from Scratch",
                color=[PAL[c] for c in cfg_order], alpha=1.0, edgecolor="white",
                hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in cfg_order], rotation=12, ha="right", fontsize=9)
    ax.set_title(label, fontsize=11, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

# Legend
legend_elements = [
    Patch(facecolor="grey", alpha=0.55, label="Phase 13 — Toggle on trained model"),
    Patch(facecolor="grey", alpha=1.0, hatch="///", label="Phase 14 — Trained from scratch"),
]
axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right")
plt.tight_layout()
path_c = os.path.join(OUT_DIR, "figC_phase13_vs_phase14.png")
plt.savefig(path_c, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_c}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE D – Updated Scalability (N=4→16) with Phase 14 annotations
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure D: Scalability sweep …")

scl = pd.read_csv(os.path.join(PHASE13_DIR, "scalability_validation.csv"))
scl["N"] = scl["Config"].str.extract(r"(\d+)").astype(int)

fig = plt.figure(figsize=(18, 9))
gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.38)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1:])

fig.suptitle("SLIM v4 Scalability — Phase 14 Summary (N=4 → 16, Full System)",
             fontsize=14, fontweight="bold")

N = scl["N"].tolist()

def annotate_line(ax, N, vals, fmt=".1f", offset=8):
    for x, y in zip(N, vals):
        ax.annotate(f"{y:{fmt}}", (x, y),
                    textcoords="offset points", xytext=(0, offset),
                    ha="center", fontsize=9)

# Success rate
ax0.plot(N, scl["Success %"], "o-", color=PAL["full_system"], lw=2.5, ms=8)
ax0.fill_between(N, scl["Success %"], alpha=0.12, color=PAL["full_system"])
ax0.set_title("Trade Success Rate (%)", fontsize=11)
ax0.set_xlabel("# Agents"); ax0.set_ylabel("%")
annotate_line(ax0, N, scl["Success %"].tolist())

# Grid dependency
ax1.plot(N, scl["Grid %"], "s-", color=PAL["no_alignment"], lw=2.5, ms=8)
ax1.fill_between(N, scl["Grid %"], alpha=0.12, color=PAL["no_alignment"])
ax1.set_title("Grid Dependency (%)", fontsize=11)
ax1.set_xlabel("# Agents"); ax1.set_ylabel("%")
annotate_line(ax1, N, scl["Grid %"].tolist())

# P2P volume (log)
ax2.semilogy(N, scl["P2P Vol"], "^-", color=PAL["no_curriculum"], lw=2.5, ms=8)
ax2.set_title("P2P Volume (kWh/step, log)", fontsize=11)
ax2.set_xlabel("# Agents"); ax2.set_ylabel("kWh")
for x, y in zip(N, scl["P2P Vol"].tolist()):
    ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8.5)

# Carbon
ax3.bar(N, scl["Carbon (kg)"] / 1000, color=["#AAB7B8","#7F8C8D","#566573","#2C3E50"], width=1.5, edgecolor="w")
ax3.set_title("Carbon Emissions (tonnes CO₂)", fontsize=11)
ax3.set_xlabel("# Agents"); ax3.set_ylabel("tonnes")
ax3.set_xticks(N)
for x, y in zip(N, (scl["Carbon (kg)"] / 1000).tolist()):
    ax3.text(x, y + 0.3, f"{y:.1f}t", ha="center", fontsize=9)

# Liquidity growth (annotated)
base = scl["P2P Vol"].iloc[0]
mults = (scl["P2P Vol"] / base).tolist()
ax4.plot(N, scl["P2P Vol"], "o-", color="#27AE60", lw=3, ms=10, zorder=3)
ax4.fill_between(N, scl["P2P Vol"], alpha=0.15, color="#27AE60")
for xi, yi, m in zip(N, scl["P2P Vol"].tolist(), mults):
    ax4.annotate(f"{m:.1f}×\n({yi:.4f} kWh)",
                 xy=(xi, yi), xytext=(xi, yi * 1.20),
                 ha="center", fontsize=9,
                 arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))
ax4.set_title("Market Liquidity — Network Effect (vs N=4 baseline)", fontsize=11)
ax4.set_xlabel("Number of Agents"); ax4.set_ylabel("P2P Volume (kWh/step)")
ax4.set_xticks(N); ax4.set_xticklabels([f"N={n}" for n in N], fontsize=11)
ax4.grid(axis="y", linestyle="--", alpha=0.3)

path_d = os.path.join(OUT_DIR, "figD_scalability_updated.png")
plt.savefig(path_d, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_d}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE E – Summary dashboard (1-page overview)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure E: Summary dashboard …")

fig = plt.figure(figsize=(20, 11))
gs  = gridspec.GridSpec(2, 4, hspace=0.55, wspace=0.42)
fig.suptitle("SLIM v4 — Complete Results Dashboard (Phase 14)",
             fontsize=16, fontweight="bold", y=1.01)

# --- Panel 1: Success rate comparison (P13 vs P14) ---
ax = fig.add_subplot(gs[0, 0])
p13_succ = [p13.loc[c, "Success %"] for c in cfg_order]
p14_succ = [p14.loc[c, "success"] * 100 for c in cfg_order]
x = np.arange(3)
ax.bar(x - 0.2, p13_succ, 0.35, color=[PAL[c] for c in cfg_order], alpha=0.55, label="Ph13 Toggle")
ax.bar(x + 0.2, p14_succ, 0.35, color=[PAL[c] for c in cfg_order], alpha=1.0, hatch="///", label="Ph14 Scratch")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c].replace(" ", "\n") for c in cfg_order], fontsize=8)
ax.set_title("Success Rate (%)\nPh13 vs Ph14", fontsize=10)
ax.legend(fontsize=7)

# --- Panel 2: Grid dependency ---
ax = fig.add_subplot(gs[0, 1])
p13_grid = [p13.loc[c, "Grid %"] for c in cfg_order]
p14_grid = [p14.loc[c, "grid_dep"] * 100 for c in cfg_order]
ax.bar(x - 0.2, p13_grid, 0.35, color=[PAL[c] for c in cfg_order], alpha=0.55)
ax.bar(x + 0.2, p14_grid, 0.35, color=[PAL[c] for c in cfg_order], alpha=1.0, hatch="///")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c].replace(" ", "\n") for c in cfg_order], fontsize=8)
ax.set_title("Grid Dependency (%)\nPh13 vs Ph14", fontsize=10)

# --- Panel 3: P2P Volume ---
ax = fig.add_subplot(gs[0, 2])
p13_p2p = [p13.loc[c, "P2P Vol"] for c in cfg_order]
p14_p2p = [p14.loc[c, "p2p_volume"] for c in cfg_order]
ax.bar(x - 0.2, p13_p2p, 0.35, color=[PAL[c] for c in cfg_order], alpha=0.55)
ax.bar(x + 0.2, p14_p2p, 0.35, color=[PAL[c] for c in cfg_order], alpha=1.0, hatch="///")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c].replace(" ", "\n") for c in cfg_order], fontsize=8)
ax.set_title("P2P Volume (kWh/step)\nPh13 vs Ph14", fontsize=10)

# --- Panel 4: Economic Profit ---
ax = fig.add_subplot(gs[0, 3])
p14_econ = [p14.loc[c, "economic_profit"] for c in cfg_order]
colors_ep = [PAL[c] for c in cfg_order]
bars = ax.barh([LABELS[c] for c in cfg_order], p14_econ, color=colors_ep, edgecolor="w")
ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_title("Economic Profit ($/step)\nPhase 14 Scratch", fontsize=10)
for bar, v in zip(bars, p14_econ):
    ax.text(v - 0.001, bar.get_y() + bar.get_height()/2,
            f"${v:.4f}", va="center", ha="right", fontsize=8)

# --- Panel 5: Scalability line ---
ax = fig.add_subplot(gs[1, :2])
ax.plot(N, scl["Success %"], "o-", color=PAL["full_system"], lw=2.5, ms=8, label="Success %")
ax2_twin = ax.twinx()
ax2_twin.semilogy(N, scl["P2P Vol"], "^--", color=PAL["no_curriculum"], lw=2, ms=7, label="P2P Vol (log)")
ax.set_xlabel("# Agents"); ax.set_ylabel("Success Rate (%)", color=PAL["full_system"])
ax2_twin.set_ylabel("P2P Volume (kWh/step, log)", color=PAL["no_curriculum"])
ax.set_xticks(N)
ax.set_title("Scalability: N=4 → 16 (Full System)", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
ax2_twin.legend(loc="lower right", fontsize=8)

# --- Panel 6: Convergence snippet ---
ax = fig.add_subplot(gs[1, 2:])
for cfg, color in PAL.items():
    sub = df_hist[df_hist["config"] == cfg].groupby("step")["success"]
    steps = sub.mean().index.tolist()
    mean  = sub.mean().values * 100
    std   = sub.std(ddof=0).fillna(0).values * 100
    ax.plot(steps, mean, label=LABELS[cfg], color=color, lw=2.2)
    ax.fill_between(steps, mean-std, mean+std, alpha=0.13, color=color)
ax.set_title("Convergence: Success Rate During Training\n(From Scratch, 3 seeds)", fontsize=10)
ax.set_xlabel("Training Step"); ax.set_ylabel("Success Rate (%)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)

path_e = os.path.join(OUT_DIR, "figE_summary_dashboard.png")
plt.savefig(path_e, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {path_e}")

print(f"\n✅  All Phase 14 plots saved to: {OUT_DIR}")
print(f"   figA — Convergence curves (5 metrics)")
print(f"   figB — Final bars (from-scratch)")
print(f"   figC — Phase 13 vs Phase 14 head-to-head")
print(f"   figD — Scalability sweep (annotated)")
print(f"   figE — Full dashboard (publication-ready)")
