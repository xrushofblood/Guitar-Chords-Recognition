import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# --- NEW FUNCTION: SAVE CONFUSION MATRIX ---
def save_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Creates and saves a visual heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 9))
    
    # annot=True writes the numbers, fmt='d' ensures they are integers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Guitar Chord Recognition', fontsize=15)
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Confusion Matrix saved to: {output_path}")

def train_guitar_model():
    # 1. SETUP PATHS
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_path = os.path.join(project_root, "data", "extracted_features", "chord_features_clean.csv")
    model_save_path = os.path.join(project_root, "models", "guitar_chord_rf_model.pkl")
    matrix_save_path = os.path.join(project_root, "models", "confusion_matrix.png")

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # 2. LOAD DATASET
    df = pd.read_csv(data_path)
    df['group'] = df['filename'].apply(lambda x: "_".join(str(x).split('_')[:3]))
    
    X = df.drop(columns=['filename', 'label', 'group'])
    y = df['label']
    groups = df['group']

    # 3. DEFINE MODEL (Using your Winning Parameters)
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features='log2',
        criterion='gini',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 4. BALANCED TRAIN/TEST SPLIT (Group-Aware)
    # Using random_state=15 as per your last attempt
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=15)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # 5. INTERNAL CROSS-VALIDATION
    print("\n[STEP] Performing Internal Cross-Validation...")
    sgkf = StratifiedGroupKFold(n_splits=5)
    cv_preds = cross_val_predict(rf_model, X_train, y_train, groups=groups_train, cv=sgkf)
    print(f"INTERNAL CV ACCURACY: {accuracy_score(y_train, cv_preds) * 100:.2f}%")

    # 6. FINAL EVALUATION ON TEST SET
    print("[STEP] Training on full Train set and testing on Unseen Videos...")
    rf_model.fit(X_train, y_train)
    final_preds = rf_model.predict(X_test)
    
    print("\n" + "#"*50)
    print(f"FINAL TEST ACCURACY: {accuracy_score(y_test, final_preds) * 100:.2f}%")
    print("#"*50)
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds, zero_division=0))

    # --- 7. NEW: GENERATE CONFUSION MATRIX IMAGE ---
    all_labels = sorted(y.unique())
    save_confusion_matrix(y_test, final_preds, all_labels, matrix_save_path)

    # 8. PRODUCTION TRAINING & SAVING
    print(f"\n[STEP] Saving final production model...")
    rf_model.fit(X, y) # Train on 100% of data for the final file
    joblib.dump(rf_model, model_save_path)
    print("Process completed successfully!")

if __name__ == "__main__":
    train_guitar_model()