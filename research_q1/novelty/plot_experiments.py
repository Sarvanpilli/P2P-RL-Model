
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_data(log_dir, scalar_tag='rollout/ep_rew_mean'):
    """
    Extracts scalar data from TensorBoard log files.
    """
    data = []
    # Find all event files in subdirectories
    event_files = glob.glob(os.path.join(log_dir, "**/*.tfevents*"), recursive=True)
    
    for ef in event_files:
        try:
            ea = EventAccumulator(ef)
            ea.Reload()
            
            # Determine run name from parent folder
            run_name = os.path.basename(os.path.dirname(ef))
            
            if scalar_tag in ea.Tags()['scalars']:
                events = ea.Scalars(scalar_tag)
                for e in events:
                    data.append({
                        'Run': run_name,
                        'Step': e.step,
                        'Value': e.value
                    })
        except Exception as e:
            print(f"Error reading {ef}: {e}")
            
    return pd.DataFrame(data)

def plot_learning_curves():
    print("Generating Learning Curves...")
    
    # Define log directories
    log_dirs = [
        "research_q1/logs/slim_ppo_tensorboard",       # N=4 Baseline
        "research_q1/logs/slim_ppo_scale_tensorboard", # Scalability
        "research_q1/logs/slim_ablation_tensorboard"  # Ablations
    ]
    
    all_data = []
    for d in log_dirs:
        if os.path.exists(d):
            df = extract_tensorboard_data(d)
            all_data.append(df)
            
    if not all_data:
        print("No data found!")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Filter/Rename for clarity
    # Map raw run names to pretty labels
    # e.g. PPO_4 -> SLIM (N=4)
    # PPO_N10 -> SLIM (N=10)
    # PPO_N4_NoSafety -> No Safety
    # PPO_N4_NoP2P -> No P2P
    
    def rename_run(name):
        if "PPO_4" in name and "No" not in name: return "SLIM (N=4)"
        if "PPO_5" in name: return "SLIM (N=4)" # Retry run
        if "N10" in name: return "SLIM (N=10)"
        if "NoSafety" in name: return "No Safety"
        if "NoP2P" in name: return "No P2P"
        return name
        
    full_df['Experiment'] = full_df['Run'].apply(rename_run)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.lineplot(data=full_df, x="Step", y="Value", hue="Experiment", alpha=0.8)
    
    plt.title("Learning Curves: Reward vs Timesteps", fontsize=14)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend(title="Experiment")
    plt.tight_layout()
    
    os.makedirs("research_q1/results", exist_ok=True)
    plt.savefig("research_q1/results/learning_curves.png", dpi=300)
    print("Saved learning_curves.png")

def plot_scalability_trend(results_dict):
    """
    results_dict: {N: {'profit': float, 'safety': float}}
    """
    print("Generating Scalability Trend...")
    
    ns = sorted(results_dict.keys())
    profits = [results_dict[n]['profit'] for n in ns]
    safety = [results_dict[n]['safety'] for n in ns]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:green'
    ax1.set_xlabel('Number of Agents (N)')
    ax1.set_ylabel('Mean Daily Profit ($)', color=color)
    ax1.plot(ns, profits, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Safety Violations', color=color)  
    ax2.plot(ns, safety, marker='x', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Scalability: Performance vs Microgrid Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("research_q1/results/scalability_trend.png", dpi=300)

def plot_ablation_bar(results_dict):
    """
    results_dict: {'ExperimentName': {'profit': float, 'safety': float}}
    """
    print("Generating Ablation Comparison...")
    
    df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    df.columns = ['Experiment', 'Profit', 'Safety']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot Profit
    sns.barplot(data=df, x='Experiment', y='Profit', palette='viridis', ax=ax)
    
    # Add Safety Labels
    for i, row in df.iterrows():
        ax.text(i, row.Profit, f"Safety Violations: {row.Safety:.1f}", 
                color='black', ha="center", va="bottom")
        
    plt.title("Ablation Study: Contribution of Components (N=4)")
    plt.ylabel("Mean Daily Profit ($)")
    plt.tight_layout()
    plt.savefig("research_q1/results/ablation_comparison.png", dpi=300)

if __name__ == "__main__":
    plot_learning_curves()
    


    plot_scalability_trend({
        4: {'profit': -1.12, 'safety': 0.0},
        10: {'profit': 5.22, 'safety': 0.0}
    })
    
    plot_ablation_bar({
        'SLIM (Baseline)': {'profit': -1.12, 'safety': 0.0},
        'No Safety': {'profit': 4.85, 'safety': 0.0},
        'No P2P': {'profit': 5.22, 'safety': 0.0}
    })

