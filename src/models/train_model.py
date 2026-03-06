import os
import pandas as pd
import numpy as np
# CHANGED: Imported StratifiedGroupKFold instead of StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_with_cross_validation():
    # Setup dynamic paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    data_path = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")
    model_dir = os.path.join(project_root, "models")
    model_path = os.path.join(model_dir, "rf_chord_classifier.pkl")

    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Cleaned CSV not found. Please run the data cleaner script first.")
        return

    # =========================================================
    # NEW: CREATE GROUPS TO PREVENT LEAKAGE
    # =========================================================
    # Extracts the video ID (e.g., "Em_Em_02") from the filename
    df['group'] = df['filename'].apply(lambda x: "_".join(str(x).split('_')[:3]))
    groups = df['group']

    # Extract features and labels. We also drop the 'group' column so it's not used as a feature!
    X = df.drop(columns=['filename', 'label', 'group'])
    y = df['label']

    print(f"Total samples loaded: {len(X)}")
    print(f"Total UNIQUE videos (groups): {groups.nunique()}") # Tells you how many real videos you have
    
    print("\nClass distribution:")
    print(y.value_counts())
    
    # Check the smallest class to avoid K-Fold crashing
    min_class_count = y.value_counts().min()
    k_folds = min(5, min_class_count)
    
    if k_folds < 2:
        print("\nError: You have a chord with only 1 sample. Cross-validation requires at least 2.")
        return

    print(f"\nInitializing {k_folds}-Fold STRATIFIED GROUP Cross-Validation...")
    print("The model is now forced to test on completely unseen videos to prevent data leakage.")
    
    # Initialize Random Forest with optimized parameters for small datasets
    
    rf_model = RandomForestClassifier(
        n_estimators=800,           # More trees for stability
        max_depth=8,               # Limit depth to prevent learning specific hand-shapes by heart
        min_samples_leaf=6,         # Each "answer" must be based on at least 4 different samples
        max_features='log2',        # Look at fewer features at a time to find more robust rules
        class_weight='balanced',
        criterion='entropy',
        random_state=42 
    )
    
    # CHANGED: StratifiedGroupKFold ensures no video is split between train and test
    cv = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # CHANGED: Passed the 'groups' parameter to cross_val_predict
    y_pred_cv = cross_val_predict(rf_model, X, y, groups=groups, cv=cv)
    
    # Calculate the real overall metrics
    overall_accuracy = accuracy_score(y, y_pred_cv)
    
    print("\n" + "="*50)
    print(f"REAL CROSS-VALIDATED ACCURACY: {overall_accuracy * 100:.2f}%")
    print("="*50 + "\n")
    
    print("Aggregated Classification Report (across all strictly isolated folds):")
    print(classification_report(y, y_pred_cv, zero_division=0))

    # Finally, train on 100% of the data to build the smartest possible final model
    print("\nTraining the final production model on 100% of the dataset...")
    rf_model.fit(X, y)
    
    # Save the model
    joblib.dump(rf_model, model_path)
    print(f"Final model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_with_cross_validation()