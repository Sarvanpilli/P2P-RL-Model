"""
SLIM v7 plotting fix:
- Remove cumulative CO2 (misleading, always increasing)
- Plot per-step and rolling CO2 vs P2P volume and grid import
- Keep cumulative plot only for profit (and optionally P2P volume)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.join('evaluation', 'evaluation_results.csv')
OUTPUT_PATH = os.path.join('evaluation', 'results', 'realtime_energy_emissions.png')
OUTPUT_CUM = os.path.join('evaluation', 'results', 'cumulative_profit_p2p.png')
OUTPUT_SCATTER = os.path.join('evaluation', 'results', 'profit_vs_co2_scatter.png')
CARBON_INTENSITY = 0.6  # kg CO2 per kWh
ROLLING_WINDOW = 24     # hours
SCATTER_WINDOW = 6      # hours for short-term smoothing

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['hour'] = df['step'] - 1
    return df

def compute_profit_and_co2(df):
    df['retail_price'] = np.where(df['hour'] % 24 >= 17, 0.50, 0.20)
    df['trading_revenue'] = df['market_price'] * df['total_export']
    df['grid_cost'] = df['retail_price'] * df['total_import']
    df['profit'] = df['trading_revenue'] - df['grid_cost']
    df['co2_per_step'] = df['total_import'] * CARBON_INTENSITY
    # Rolling CO2 (24h avg)
    df['co2_rolling'] = df['co2_per_step'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return df

def prepare_window(df, start_idx=0, window=168):
    """Select a contiguous window (default one week at 1h steps)."""
    window_df = df.iloc[start_idx:start_idx+window].copy()
    window_df['t'] = np.arange(len(window_df))
    # cumulative (for optional profit/P2P plot)
    window_df['cum_profit'] = window_df['profit'].cumsum()
    window_df['p2p_per_step'] = window_df['total_export']  # proxy for P2P volume
    window_df['cum_p2p'] = window_df['p2p_per_step'].cumsum()
    return window_df

def plot_scatter(df, out_path=OUTPUT_SCATTER):
    """Scatter of smoothed profit vs smoothed CO2, color-coded by P2P volume."""
    # Per-step metrics
    profit = df['profit']
    co2 = df['co2_per_step']
    p2p = df['p2p_per_step']

    # Rolling smoothing
    profit_s = profit.rolling(SCATTER_WINDOW, min_periods=1).mean()
    co2_s = co2.rolling(SCATTER_WINDOW, min_periods=1).mean()
    p2p_s = p2p.rolling(SCATTER_WINDOW, min_periods=1).mean()

    # Trend line (linear regression)
    import numpy as np
    mask = (~profit_s.isna()) & (~co2_s.isna())
    x = profit_s[mask].values
    y = co2_s[mask].values
    if len(x) > 1:
        a, b = np.polyfit(x, y, 1)
    else:
        a, b = 0.0, np.mean(y) if len(y) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=p2p_s[mask].values, cmap='viridis', alpha=0.7, label='Intervals')
    ax.plot(x, a*x + b, color='red', linestyle='--', label=f'Trend: CO2 = {a:.3f}*Profit + {b:.2f}')
    cb = plt.colorbar(sc, ax=ax, label='P2P Volume (smoothed)')
    ax.set_title('Profit vs CO₂ Emissions (Short-Term Dynamics)')
    ax.set_xlabel('Profit per Interval ($)')
    ax.set_ylabel('CO₂ Emissions per Interval (kg)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f'Saved scatter plot -> {out_path}')
    plt.close()

def plot_realtime(df, out_path=OUTPUT_PATH, normalize=True):
    """Plot per-step P2P, grid import, and 24h rolling CO2."""
    p2p = df['p2p_per_step'].values
    grid = df['total_import'].values
    co2r = df['co2_rolling'].values
    t = df['t'].values

    if normalize:
        def norm(x):
            m = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
            return x / m
        p2p_n = norm(p2p)
        grid_n = norm(grid)
        co2r_n = norm(co2r)
    else:
        p2p_n, grid_n, co2r_n = p2p, grid, co2r

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, p2p_n, label='P2P Volume (kWh)', color='teal')
    ax.plot(t, grid_n, label='Grid Import (kWh)', color='firebrick')
    ax.plot(t, co2r_n, label='CO₂ Emissions (24h avg)', color='darkorange', linestyle='--')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Normalized value' if normalize else 'Value')
    ax.set_title('Real-Time Energy & Emissions Dynamics (SLIM v7)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f'Saved plot -> {out_path}')
    plt.close()

def plot_cumulative(df, out_path=OUTPUT_CUM):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df['t'], df['cum_profit'], label='Cumulative Profit', color='navy')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Cumulative Profit ($)')
    ax2 = ax1.twinx()
    ax2.plot(df['t'], df['cum_p2p'], label='Cumulative P2P Volume', color='seagreen', linestyle='--')
    ax2.set_ylabel('Cumulative P2P (kWh)')
    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc='upper left')
    ax1.set_title('Cumulative Profit and P2P Volume (No CO₂)')
    ax1.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f'Saved cumulative plot -> {out_path}')
    plt.close()


def main():
    df = load_data()
    df = compute_profit_and_co2(df)
    window_df = prepare_window(df, start_idx=0, window=168)  # default: 1 week
    plot_realtime(window_df, OUTPUT_PATH, normalize=True)
    plot_cumulative(window_df, OUTPUT_CUM)

if __name__ == '__main__':
    main()
