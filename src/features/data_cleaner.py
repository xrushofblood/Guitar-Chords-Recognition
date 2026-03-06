import pandas as pd
import numpy as np
import os

def clean_dataset(input_csv, output_csv):
    print("Starting dataset cleaning...")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    original_len = len(df)
    feature_cols = [f'cell_{i}' for i in range(18)]

    # Ensure data is numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_cols)

    # 1. Remove Exact Duplicates
    df = df.drop_duplicates(subset=feature_cols, keep='first')
    after_dups = len(df)

    # 2. Remove "Empty Grids" (Missed Chords)
    is_chord = df['label'] != 'N'
    is_empty = df[feature_cols].sum(axis=1) < 0.15  
    df = df[~(is_chord & is_empty)]
    after_empty = len(df)

    # 3. Remove Outliers (Statistical Distance)
    clean_df = pd.DataFrame()

    for label in df['label'].unique():
        group = df[df['label'] == label]
        
        if label == 'N':
            clean_df = pd.concat([clean_df, group])
            continue
            
        if len(group) >= 4:
            mean = group[feature_cols].mean()
            std = group[feature_cols].std(ddof=0).replace(0, 1e-6)
            z_scores = np.abs((group[feature_cols] - mean) / std)
            
            # Keep rows within 2.5 std devs
            filtered_group = group[(z_scores < 2.5).all(axis=1)]
            clean_df = pd.concat([clean_df, filtered_group])
        else:
            clean_df = pd.concat([clean_df, group])

    final_len = len(clean_df)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    print("\n--- CLEANING REPORT ---")
    print(f"Initial samples:              {original_len}")
    print(f"Duplicates removed:           {original_len - after_dups}")
    print(f"Empty grids removed:          {after_dups - after_empty}")
    print(f"Statistical outliers removed: {after_empty - final_len}")
    print(f"-------------------------------")
    print(f"FINAL CLEAN SAMPLES:          {final_len}")
    print(f"File saved as: {output_csv}")

if __name__ == "__main__":
    # Calculate absolute paths dynamically based on this file's location
    # This file is in src/features/, so we go up two levels to reach the root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    INPUT_FILE = os.path.join(project_root, "data", "extracted_features", "chord_features.csv")
    OUTPUT_FILE = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")
    
    if os.path.exists(INPUT_FILE):
        clean_dataset(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"File not found: {INPUT_FILE}")
        print("Make sure the path is correct relative to the project root.")