import pandas as pd
import os
import numpy as np

def smart_data_cleaning():
    # 1. SETUP PATHS
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    input_path = os.path.join(project_root, "data", "extracted_features", "chord_features.csv")
    output_path = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading raw features from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. SEPARATE CHORDS FROM NULLS
    df_chords = df[df['label'] != 'N'].copy()
    df_nulls = df[df['label'] == 'N'].copy()

    print(f"Initial Chords samples: {len(df_chords)}")
    print(f"Initial Null samples:   {len(df_nulls)}")

    # 3. SMART NULL SELECTION LOGIC
    # Categorize Nulls into 'Pure Empty' and 'Transition' based on filename
    # N_NULL_01... -> Pure Empty
    # N_A_01...    -> Transition from chord A
    
    def categorize_null(filename):
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1] # Returns 'NULL', 'A', 'Am', 'Dm', etc.
        return 'UNKNOWN'

    df_nulls['source_chord'] = df_nulls['filename'].apply(categorize_null)

    selected_nulls_list = []

    # Strategy A: Keep ALL "Pure Empty" guitar frames (N_NULL_...)
    pure_empty = df_nulls[df_nulls['source_chord'] == 'NULL']
    selected_nulls_list.append(pure_empty)
    print(f"  - Keeping all {len(pure_empty)} pure empty guitar frames (N_NULL).")

    # Strategy B: Sample proportionally from chord transitions (N_A, N_Am, etc.)
    transition_nulls = df_nulls[df_nulls['source_chord'] != 'NULL']
    transition_groups = transition_nulls.groupby('source_chord')

    # We want a diverse set of transitions. 
    # We will take roughly 10-12 samples per transition type to keep it balanced.
    samples_per_transition = 10 

    for chord_name, group in transition_groups:
        n_to_take = min(len(group), samples_per_transition)
        sampled_group = group.sample(n=n_to_take, random_state=42)
        selected_nulls_list.append(sampled_group)
        print(f"  - Sampled {n_to_take} frames from '{chord_name}' transitions.")

    # 4. RECOMBINE AND CLEAN UP
    df_final_nulls = pd.concat(selected_nulls_list).drop(columns=['source_chord'])
    
    # Final Dataset
    df_clean = pd.concat([df_chords, df_final_nulls])
    
    # Optional: Final shuffle
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "="*30)
    print("FINAL CLEANED DISTRIBUTION:")
    print(df_clean['label'].value_counts())
    print("="*30)

    # 5. SAVE
    df_clean.to_csv(output_path, index=False)
    print(f"Clean dataset saved to: {output_path}")

if __name__ == "__main__":
    smart_data_cleaning()