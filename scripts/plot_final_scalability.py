
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the fixed data
df = pd.read_csv("research_q1/results/scalability_v5/scalability_metrics_v5_FIXED.csv")

# Plotting
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Panel 1: P2P Volume Growth
plt.subplot(1, 2, 1)
bars = plt.bar(df['Agents'].astype(str), df['P2P (kWh)'], color='#1D9E75', alpha=0.8)
plt.title("Total Network Liquidity surge (N=4 to 24)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Agents (N)", fontsize=12)
plt.ylabel("Total P2P Volume (kWh)", fontsize=12)
# Annotate values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

# Panel 2: P2P Success Rate
plt.subplot(1, 2, 2)
plt.plot(df['Agents'].astype(str), df['Success %'], marker='o', color='#378ADD', linewidth=3, markersize=10)
plt.title("Market Matching Efficiency (%)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Agents (N)", fontsize=12)
plt.ylabel("Trade Success Rate (%)", fontsize=12)
plt.ylim(0, 100)
# Annotate
for i, txt in enumerate(df['Success %']):
    plt.annotate(f"{txt:.1f}%", (i, df['Success %'][i]+2), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("C:/Users/Sarva/.gemini/antigravity/brain/118aca90-744c-4222-a7c1-aaf2f0fd8939/figD_scalability_final.png", dpi=300)
plt.close()

print("Final scalability plot saved.")
