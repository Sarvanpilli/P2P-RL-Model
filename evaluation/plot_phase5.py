"""
Phase 5 Evaluation Visualization Script

Creates comprehensive visualizations for Phase 5 model performance analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_results():
    """Load all evaluation results"""
    results = {}
    
    checkpoints = ["50k", "100k", "150k", "200k", "250k", "baseline"]
    
    for cp in checkpoints:
        file_path = f"evaluation/results_phase5_{cp}.csv"
        if os.path.exists(file_path):
            results[cp] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    return results

def plot_comprehensive_analysis(results):
    """Create comprehensive multi-panel analysis plot"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Phase 5 Hybrid Model: Comprehensive Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    checkpoints = ["50k", "100k", "150k", "200k", "250k", "baseline"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Mean Reward Comparison
    ax1 = axes[0, 0]
    mean_rewards = [results[cp]['total_reward'].mean() for cp in checkpoints if cp in results]
    ax1.bar(range(len(mean_rewards)), mean_rewards, color=colors[:len(mean_rewards)])
    ax1.set_xticks(range(len(checkpoints)))
    ax1.set_xticklabels(checkpoints, rotation=45)
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Reward by Checkpoint')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Reward Over Time
    ax2 = axes[0, 1]
    for i, cp in enumerate(checkpoints):
        if cp in results:
            cumulative = results[cp]['total_reward'].cumsum()
            ax2.plot(cumulative, label=cp, color=colors[i], linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Reward Progression')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Grid Import Comparison
    ax3 = axes[1, 0]
    total_imports = [results[cp]['total_import'].sum() for cp in checkpoints if cp in results]
    ax3.bar(range(len(total_imports)), total_imports, color=colors[:len(total_imports)])
    ax3.set_xticks(range(len(checkpoints)))
    ax3.set_xticklabels(checkpoints, rotation=45)
    ax3.set_ylabel('Total Grid Import (kWh)')
    ax3.set_title('Grid Import by Checkpoint')
    ax3.grid(True, alpha=0.3)
    
    # 4. Battery SoC Profile
    ax4 = axes[1, 1]
    for i, cp in enumerate(checkpoints):
        if cp in results:
            ax4.plot(results[cp]['soc_mean'], label=cp, color=colors[i], linewidth=2, alpha=0.7)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Mean SoC (%)')
    ax4.set_title('Battery State of Charge Profile')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    # 5. P2P Trading Volume
    ax5 = axes[2, 0]
    p2p_volumes = [results[cp]['p2p_volume'].sum() for cp in checkpoints if cp in results]
    ax5.bar(range(len(p2p_volumes)), p2p_volumes, color=colors[:len(p2p_volumes)])
    ax5.set_xticks(range(len(checkpoints)))
    ax5.set_xticklabels(checkpoints, rotation=45)
    ax5.set_ylabel('Total P2P Volume (kWh)')
    ax5.set_title('P2P Trading Volume by Checkpoint')
    ax5.grid(True, alpha=0.3)
    
    # 6. Net Grid Flow Over Time (250k vs Baseline)
    ax6 = axes[2, 1]
    if "250k" in results:
        ax6.plot(results["250k"]['net_grid_flow'], 
                label='250k Checkpoint', color='#d62728', linewidth=2)
    if "baseline" in results:
        ax6.plot(results["baseline"]['net_grid_flow'], 
                label='Baseline', color='#8c564b', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Net Grid Flow (kWh)')
    ax6.set_title('Net Grid Flow: 250k vs Baseline')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation/phase5_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: evaluation/phase5_comprehensive_analysis.png")
    plt.close()

def plot_learning_progression():
    """Plot learning progression metrics"""
    
    summary = pd.read_csv("evaluation/phase5_evaluation_summary.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 5 Training Progression Analysis', fontsize=16, fontweight='bold')
    
    # Filter out baseline for progression plots
    rl_data = summary[summary['checkpoint'] != 'Baseline'].copy()
    
    # Convert checkpoint names to numeric steps
    step_map = {'50k': 50, '100k': 100, '150k': 150, '200k': 200, '250k': 250}
    rl_data['steps_k'] = rl_data['checkpoint'].map(step_map)
    rl_data = rl_data.sort_values('steps_k')
    
    # 1. Mean Reward Progression
    ax1 = axes[0, 0]
    ax1.plot(rl_data['steps_k'], rl_data['mean_reward'], 
            marker='o', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_xlabel('Training Steps (thousands)')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Reward Progression During Training')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Grid Import Progression
    ax2 = axes[0, 1]
    ax2.plot(rl_data['steps_k'], rl_data['total_import'], 
            marker='s', linewidth=2, markersize=8, color='#ff7f0e')
    ax2.set_xlabel('Training Steps (thousands)')
    ax2.set_ylabel('Total Grid Import (kWh)')
    ax2.set_title('Grid Import Reduction Over Training')
    ax2.grid(True, alpha=0.3)
    
    # 3. Battery Utilization (SoC)
    ax3 = axes[1, 0]
    ax3.plot(rl_data['steps_k'], rl_data['mean_soc'], 
            marker='^', linewidth=2, markersize=8, color='#2ca02c')
    ax3.set_xlabel('Training Steps (thousands)')
    ax3.set_ylabel('Mean SoC (%)')
    ax3.set_title('Battery Utilization Over Training')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # 4. P2P Trading Volume
    ax4 = axes[1, 1]
    ax4.plot(rl_data['steps_k'], rl_data['p2p_volume'], 
            marker='D', linewidth=2, markersize=8, color='#d62728')
    ax4.set_xlabel('Training Steps (thousands)')
    ax4.set_ylabel('P2P Trading Volume (kWh)')
    ax4.set_title('P2P Trading Activity Over Training')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation/phase5_learning_progression.png', dpi=300, bbox_inches='tight')
    print("Saved: evaluation/phase5_learning_progression.png")
    plt.close()

def plot_detailed_250k_analysis(results):
    """Detailed analysis of the 250k checkpoint"""
    
    if "250k" not in results:
        print("250k checkpoint data not found")
        return
    
    df = results["250k"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 5 @ 250k Steps: Detailed Behavioral Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Reward decomposition over time
    ax1 = axes[0, 0]
    ax1.plot(df['total_reward'], label='Total Reward', linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Signal Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. SoC trajectory
    ax2 = axes[0, 1]
    ax2.plot(df['soc_mean'], linewidth=2, color='green')
    if 'soc_std' in df.columns:
        ax2.fill_between(range(len(df)), 
                         df['soc_mean'] - df['soc_std'],
                         df['soc_mean'] + df['soc_std'],
                         alpha=0.3, color='green')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('SoC (%)')
    ax2.set_title('Battery State of Charge (Mean ± Std)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. Grid interaction
    ax3 = axes[1, 0]
    ax3.plot(df['total_import'], label='Import', linewidth=2, color='red')
    ax3.plot(df['total_export'], label='Export', linewidth=2, color='blue')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Energy (kWh)')
    ax3.set_title('Grid Import/Export Pattern')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Market price vs grid penalty
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    ax4.plot(df['market_price'], label='Market Price', 
            linewidth=2, color='purple', alpha=0.7)
    ax4_twin.plot(df['grid_penalty'], label='Grid Penalty', 
                 linewidth=2, color='orange', alpha=0.7)
    
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Market Price ($/kWh)', color='purple')
    ax4_twin.set_ylabel('Grid Penalty', color='orange')
    ax4.set_title('Market Price vs Grid Penalty')
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig('evaluation/phase5_250k_detailed.png', dpi=300, bbox_inches='tight')
    print("Saved: evaluation/phase5_250k_detailed.png")
    plt.close()

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*60)
    print("Generating Phase 5 Evaluation Visualizations")
    print("="*60 + "\n")
    
    # Load results
    results = load_results()
    
    if not results:
        print("ERROR: No evaluation results found!")
        return
    
    # Generate plots
    plot_comprehensive_analysis(results)
    plot_learning_progression()
    plot_detailed_250k_analysis(results)
    
    print("\n" + "="*60)
    print("Visualization Generation Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
