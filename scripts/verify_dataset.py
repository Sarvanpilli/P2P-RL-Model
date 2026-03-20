import pandas as pd
import os
import numpy as np

def verify():
    print("Dataset Verification Script...")
    
    hybrid_path = "processed_hybrid_data.csv"
    ausgrid_path = "evaluation/ausgrid_p2p_energy_dataset.csv"
    output_path = "evaluation/dataset_verification.txt"
    
    if not os.path.exists(hybrid_path):
        print(f"Error: {hybrid_path} not found.")
        return

    df = pd.read_csv(hybrid_path)
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    if 'timestamp' in df.columns:
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\nColumn Statistics (Min/Max):")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}")
    
    # Verdict logic
    verdict = ""
    if os.path.exists(ausgrid_path):
        df_aus = pd.read_csv(ausgrid_path)
        if len(df) == len(df_aus):
            verdict = "VERDICT: This is REAL data"
        else:
            verdict = "VERDICT: This is SYNTHETIC data"
    else:
        # Fallback if ausgrid file missing
        if len(df) == 8760:
            verdict = "VERDICT: This is REAL data"
        else:
            verdict = "VERDICT: This is SYNTHETIC data"

    print(f"\n{verdict}")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(f"Dataset Verification Report\n")
        f.write(f"===========================\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        if 'timestamp' in df.columns:
            f.write(f"Span: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"{verdict}\n")

if __name__ == "__main__":
    verify()
