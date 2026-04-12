"""
Generate a correct "cumulative profit vs CO2" plot from evaluation_results.csv.
Profit = trading_revenue - grid_cost (no reward penalties).
CO2 = grid_import * carbon_intensity.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = os.path.join('evaluation', 'evaluation_results.csv')
OUTPUT_PATH = os.path.join('evaluation', 'results', 'cumulative_profit_co2.png')
CARBON_INTENSITY = 0.6  # kg CO2 per kWh

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['hour'] = df['step'] - 1
    return df

def compute_profit_and_co2(df):
    df['retail_price'] = np.where(df['hour'] % 24 >= 17, 0.50, 0.20)
    df['trading_revenue'] = df['market_price'] * df['total_export']
    df['grid_cost'] = df['retail_price'] * df['total_import']
    df['profit'] = df['trading_revenue'] - df['grid_cost']
    df['co2'] = df['total_import'] * CARBON_INTENSITY
    return df

def prepare_window(df, start_hour=0, window=24, ma=5):
    window_df = df[(df['hour'] >= start_hour) & (df['hour'] < start_hour + window)].copy()
    window_df['profit_smooth'] = window_df['profit'].rolling(ma, min_periods=1).mean()
    window_df['co2_smooth'] = window_df['co2'].rolling(ma, min_periods=1).mean()
    window_df['cumulative_profit'] = window_df['profit_smooth'].cumsum()
    window_df['cumulative_co2'] = window_df['co2_smooth'].cumsum()
    window_df['t'] = np.arange(len(window_df))
    return window_df

def plot(df, out_path=OUTPUT_PATH):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df['t'], df['cumulative_profit'], label='Cumulative Profit')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Cumulative Profit ($)')

    ax2 = ax1.twinx()
    ax2.plot(df['t'], df['cumulative_co2'], label='Cumulative CO2', linestyle='--')
    ax2.set_ylabel('Cumulative CO2 (kg)')

    for x, label in [(0, 'Early (0-8h)'), (9, 'Midday (9-17h)'), (18, 'Evening (18-24h)')]:
        ax1.axvline(x=x, color='gray', linestyle=':', linewidth=1)
        ax1.text(x + 0.2, ax1.get_ylim()[1] * 0.95, label, fontsize=9, color='gray')

    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc='upper left')
    plt.title('Economic vs Environmental Performance Over Time')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f'Saved plot -> {out_path}')


def main():
    df = load_data()
    df = compute_profit_and_co2(df)
    window_df = prepare_window(df, start_hour=0, window=24, ma=5)
    plot(window_df, OUTPUT_PATH)

if __name__ == '__main__':
    main()
