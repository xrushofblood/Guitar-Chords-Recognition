import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def clean_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    input_csv = os.path.join(project_root, "data", "extracted_features", "chord_features.csv")
    output_csv = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # 1. Rimuoviamo i frame completamente vuoti (tutti zeri)
    df = df[(df[feature_cols] != 0).any(axis=1)]

    cleaned_dfs = []
    
    # 2. Outlier Detection e Bilanciamento
    for chord_label, group in df.groupby('label'):
        # Pulizia Outliers (Isolation Forest)
        if len(group) > 10:
            clf = IsolationForest(contamination=0.05, random_state=42)
            preds = clf.fit_predict(group[feature_cols])
            group = group[preds == 1]

        # --- SMART BALANCING ---
        if chord_label == 'N' and len(group) > 35:
            group = group.sample(n=35, random_state=42) 
            print(f"  - [{chord_label}]: Downsampled to 50 samples for balancing.")
        else:
            print(f"  - [{chord_label}]: Kept {len(group)} samples.")

        cleaned_dfs.append(group)

    final_df = pd.concat(cleaned_dfs, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\nCleaned & Balanced dataset saved! Total rows: {len(final_df)}")

if __name__ == "__main__":
    clean_data()