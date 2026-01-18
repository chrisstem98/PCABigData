from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# =====================
# Paths & data
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")
y_train = np.loadtxt(BASE_DIR / "train" / "y_train.txt", dtype=int)

# =====================
# Pipeline
# =====================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(max_iter=40000))
])

# =====================
# Parameter grid
# =====================
param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__loss": ["hinge", "squared_hinge"]
}

# =====================
# Cross-validation
# =====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV for Linear SVM (TRAIN ONLY)...")
grid.fit(X_train, y_train)

# =====================
# Results
# =====================
print("\nBest parameters:")
print(grid.best_params_)

print("\nBest mean CV F1:")
print(grid.best_score_)

results_df = pd.DataFrame(grid.cv_results_) \
    .sort_values("mean_test_score", ascending=False)

out_path = PROJECT_DIR / "q1_linear_svm_gridsearch_results.csv"
results_df.to_csv(out_path, index=False)
print("\nSaved full grid results to:", out_path)
