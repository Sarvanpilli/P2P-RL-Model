import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_log_dir(log_root="logs"):
    """Finds the latest PPO log directory."""
    if not os.path.exists(log_root):
        log_root = "tb_logs"
    if not os.path.exists(log_root):
        return None
    
    subdirs = [os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d)) and "PPO" in d]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)

def parse_tensorboard(log_dir):
    """Parses tensorboard event files."""
    if not log_dir: return {}
    
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files: return {}
    
    event_file = max(event_files, key=os.path.getsize)
    print(f"Parsing TB Log: {event_file}")
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()['scalars']
    
    data = {}
    target_tags = {
        'rollout/ep_rew_mean': 'Mean Episodic Reward',
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss'
    }
    
    for tag, label in target_tags.items():
        if tag in tags:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})
            
    return data

def plot_combined_metrics(tb_data, csv_path, output_path):
    """Plots whatever metrics are available."""
    
    # Check for CSV data
    df_eval = None
    if os.path.exists(csv_path):
        try:
            df_eval = pd.read_csv(csv_path)
            # Assuming 'episode' and 'total_reward' columns
            if 'total_reward' not in df_eval.columns:
                df_eval = None
        except:
            df_eval = None

    # Determine subplots
    plots = []
    
    if 'rollout/ep_rew_mean' in tb_data:
        plots.append(('rollout/ep_rew_mean', 'Training Reward', tb_data['rollout/ep_rew_mean']))
    elif df_eval is not None:
        plots.append(('eval_reward', 'Evaluation Reward', df_eval))
    
    if 'train/value_loss' in tb_data:
        plots.append(('train/value_loss', 'Training Value Loss', tb_data['train/value_loss']))
        
    if not plots:
        print("No data found to plot.")
        return

    fig, axes = plt.subplots(len(plots), 1, figsize=(10, 6 * len(plots)))
    if len(plots) == 1: axes = [axes]
    
    for ax, (key, title, df) in zip(axes, plots):
        if key == 'eval_reward':
            ax.plot(df['episode'], df['total_reward'], label='Total Reward', color='#e74c3c', marker='o', markersize=4)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
        else:
            # TensorBoard data
            ax.plot(df['step'], df['value'], label=title, color='#3498db')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Value')
            
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    log_dir = get_latest_log_dir()
    tb_data = parse_tensorboard(log_dir)
    
    csv_path = "models/eval_metrics_summary.csv"
    output_path = "results/training_learning_curve.png"
    
    plot_combined_metrics(tb_data, csv_path, output_path)

if __name__ == "__main__":
    main()
