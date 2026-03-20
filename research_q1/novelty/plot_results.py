import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")

def plot_all_results(results_csv="research_q1/results/results_all_experiments.csv", 
                     trace_csv="research_q1/results/results_traces.csv"):
                         
    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} not found.")
        return

    df = pd.read_csv(results_csv)
    
    # 1. Separate core experiments from delta sweep
    core_exps = ["baseline_grid", "auction_old", "new_market", "no_p2p_reward", "gnn_model"]
    df_core = df[df['experiment_name'].isin(core_exps)].copy()
    
    # Identify delta sweep items
    df_delta = df[df['experiment_name'].str.startswith('delta_sweep') | (df['experiment_name'] == 'new_market')].copy()
    
    # Rename labels for nicer plotting
    labels = {
        "baseline_grid": "Grid-Only\n(No P2P)",
        "auction_old": "Old Auction\n(Static price)",
        "new_market": "SLIM P2P\n(Dynamic)",
        "no_p2p_reward": "No Spec. Reward\n(SLIM P2P)",
        "gnn_model": "GNN + SLIM\n(Dynamic)"
    }
    df_core['plot_label'] = df_core['experiment_name'].map(labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ==========================================
    # Plot 1: Bar chart: Profit comparison
    # ==========================================
    ax1 = axes[0, 0]
    sns.barplot(data=df_core, x='plot_label', y='profit', ax=ax1, capsize=0.1, errorbar='sd')
    ax1.set_title("1. Profit Comparison By Market Design", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Total Episode Profit ($)", fontsize=12)
    ax1.set_xlabel("")
    
    # ==========================================
    # Plot 2: Line plot: P2P volume vs delta
    # ==========================================
    ax2 = axes[0, 1]
    df_delta = df_delta.sort_values(by='delta')
    sns.lineplot(data=df_delta, x='delta', y='p2p_volume', ax=ax2, marker='o', markersize=8, errorbar='sd')
    ax2.set_title("2. Agent P2P Engagement vs Delta Spread", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Total P2P Volume (kWh)", fontsize=12)
    ax2.set_xlabel("Market Spread Delta", fontsize=12)
    ax2.set_xticks(sorted(df_delta['delta'].unique()))
    
    # ==========================================
    # Plot 3: Line plot: P2P volume vs timestep
    # ==========================================
    ax3 = axes[1, 0]
    try:
        if os.path.exists(trace_csv) and os.path.getsize(trace_csv) > 0:
            df_trace = pd.read_csv(trace_csv)
            if not df_trace.empty:
                df_trace['time_bin'] = (df_trace['timestep'] // 24) * 24
                sns.lineplot(data=df_trace, x='time_bin', y='p2p_volume', hue='experiment_name', ax=ax3, errorbar=None)
                ax3.set_title("3. P2P Trading Evolution Over Time", fontsize=14, fontweight='bold')
                ax3.set_ylabel("P2P Volume (kWh)", fontsize=12)
                ax3.set_xlabel("Timestep (Hour)", fontsize=12)
                ax3.legend(title="Model", title_fontsize='13', fontsize='11')
            else:
                ax3.text(0.5, 0.5, "Trace data empty", ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "Trace CSV not found", ha='center', va='center', transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Trace error: {e}", ha='center', va='center', transform=ax3.transAxes, fontsize=8)
        
    # ==========================================
    # Plot 4: Bar chart: Grid usage comparison
    # ==========================================
    ax4 = axes[1, 1]
    # To plot Import and Export side by side, melt the dataframe
    df_grid = pd.melt(df_core, id_vars=['plot_label'], value_vars=['grid_import', 'grid_export'], 
                      var_name='Grid Flow', value_name='kWh')
    df_grid['Grid Flow'] = df_grid['Grid Flow'].map({"grid_import": "Import", "grid_export": "Export"})
    
    sns.barplot(data=df_grid, x='plot_label', y='kWh', hue='Grid Flow', ax=ax4, capsize=0.1, errorbar='sd')
    ax4.set_title("4. Grid Dependency Comparison", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Energy Volume (kWh)", fontsize=12)
    ax4.set_xlabel("")
    ax4.legend(title="")
    
    plt.tight_layout()
    out_file = "research_q1/results/paper_plots.png"
    plt.savefig(out_file, dpi=300)
    print(f"Publication plots successfully saved to {out_file}")

if __name__ == "__main__":
    plot_all_results()
