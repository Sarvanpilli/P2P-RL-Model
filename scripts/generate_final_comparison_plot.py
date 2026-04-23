
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "research_q1/results/synthetic_eval/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data from evaluation_report.md (Deficit N=4)
systems = ['Rule-Based', 'SLIM-v5', 'SLIM-v7 (Emergence)']
success_rates = [0.06, 0.01, 2.75]
profits = [-16.07, -3.88, -3.88]
p2p_volumes = [0.25, 0.01, 2.74]

x = np.arange(len(systems))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Axis: Success Rate
color = 'tab:blue'
bars1 = ax1.bar(x - width/2, success_rates, width, label='Market Success Rate %', color=color, alpha=0.7, edgecolor='black')
ax1.set_xlabel('System Configuration', fontweight='bold')
ax1.set_ylabel('Success Rate (%)', color=color, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(x)
ax1.set_xticklabels(systems, fontweight='bold')

# Secondary Axis: P2P Volume
ax2 = ax1.twinx()
color = 'tab:green'
bars2 = ax2.bar(x + width/2, p2p_volumes, width, label='P2P Volume (kWh)', color=color, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Total P2P Volume (kWh)', color=color, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color)

# Add numeric labels on top of bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(bars1, ax1)
autolabel(bars2, ax2)

plt.title('SLIM v7 vs Baselines: Emergence Comparison (N=4 Deficit)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Combined Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.savefig(os.path.join(RESULTS_DIR, "v7e_vs_baselines_comparison.png"), dpi=300)
plt.close()

print(f"Comparison plot generated: {os.path.join(RESULTS_DIR, 'v7e_vs_baselines_comparison.png')}")
