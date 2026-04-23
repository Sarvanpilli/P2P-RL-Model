
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "research_q1/results/synthetic_eval/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_styled_table():
    # Data derived from SLIM v7 (Emergence Phase) vs Baselines (N=4 Deficit Scenario)
    data = [
        ["P2P Volume", "0 kWh", "0.25 kWh", "2.74 kWh"],
        ["Grid Import", "328.4 kWh", "326.1 kWh", "298.2 kWh"],
        ["Avg Reward", "-3.63", "-16.08", "-3.88"],
        ["Safety Violations", "0", "0", "0"]
    ]
    
    headers = ["Metric", "Grid-Only", "Rule-Based", "SLIM v7"]
    
    # Visual Style (Purple Theme)
    header_color = '#A38CFF' # Solid Purple
    row_colors = ['#E0D4FF', '#F3EFFF'] # Light / Very Light Purple
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    
    # Apply Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5) # Scale height for visibility
    
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
        else:
            # Alternating rows
            color_idx = (key[0] + 1) % 2
            cell.set_facecolor(row_colors[color_idx])
            
            # Bold the proposing system's results
            if key[1] == 3: # SLIM v7 Column
                cell.set_text_props(weight='bold')
                
    plt.title("SLIM v7: Cross-System Performance Benchmark", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    table_path = os.path.join(RESULTS_DIR, "scientific_comparison_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scientific styled table generated: {table_path}")

if __name__ == "__main__":
    generate_styled_table()
