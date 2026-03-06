import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def run_grid_search():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_path = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")

    print(f"Loading dataset for tuning: {data_path}")
    df = pd.read_csv(data_path)

    # 1. PREPARE DATA
    df['group'] = df['filename'].apply(lambda x: "_".join(str(x).split('_')[:3]))
    X = df.drop(columns=['filename', 'label', 'group'])
    y = df['label']
    groups = df['group']

    # 2. DEFINE PARAMETER GRID
    # We test different "shapes" for our Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    # 3. INITIALIZE GROUP CROSS-VALIDATION
    # We MUST use the same 5-Fold Group strategy to be consistent
    sgkf = StratifiedGroupKFold(n_splits=5)

    # 4. RUN GRID SEARCH
    print("Starting Grid Search... This might take a few minutes.")
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=sgkf,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1 # Uses all CPU cores
    )

    grid_search.fit(X, y, groups=groups)

    # 5. RESULTS
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print(grid_search.best_params_)
    print(f"BEST CROSS-VALIDATED ACCURACY: {grid_search.best_score_ * 100:.2f}%")
    print("="*50 + "\n")

    # Final Check: Detailed report with the best model
    best_model = grid_search.best_estimator_
    print("Note: This tuning helps the model generalize better with existing features.")

if __name__ == "__main__":
    run_grid_search()