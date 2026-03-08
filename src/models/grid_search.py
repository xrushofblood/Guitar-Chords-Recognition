import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import joblib

# 1. Load data
df = pd.read_csv("data/extracted_features/chord_features_clean.csv")
df['group'] = df['filename'].apply(lambda x: "_".join(str(x).split('_')[:3]))
X = df.drop(columns=['filename', 'label', 'group'])
y = df['label']
groups = df['group']

# 2. Define complex grid
# We test deep vs shallow, and different split qualities
param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# 3. Setup CV
sgkf = StratifiedGroupKFold(n_splits=5)
rf = RandomForestClassifier(random_state=42)

# 4. Search
print("Starting intensive Grid Search... grab a coffee, this will take time.")
grid_search = GridSearchCV(rf, param_grid, cv=sgkf, scoring='f1_macro', n_jobs=-1, verbose=2)
grid_search.fit(X, y, groups=groups)

print("\nWINNING PARAMETERS:")
print(grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")