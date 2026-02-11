from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# =====================
# Paths & data
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")
y_train = np.loadtxt(BASE_DIR / "train" / "y_train.txt", dtype=int)
X_test  = np.loadtxt(BASE_DIR / "test"  / "X_test.txt")
y_test  = np.loadtxt(BASE_DIR / "test"  / "y_test.txt", dtype=int)

# =====================
# CV setup (TRAIN ONLY)
# =====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================
# Pipeline: Scaler -> RFE -> LinearSVM
# - RFE uses an internal LinearSVC estimator to rank features
# - final classifier is another LinearSVC
# =====================
rfe_estimator = LinearSVC(max_iter=50000)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rfe", RFE(estimator=rfe_estimator, n_features_to_select=26, step=0.1)),
    ("svm", LinearSVC(max_iter=50000))
])

# =====================
# Grid search (TRAIN ONLY)
# We tune the final SVM hyperparameters (svm__*)
# Optional: you can also tune the RFE estimator's C via rfe__estimator__C
# =====================
param_grid = {
    "svm__C": [0.01, 0.1, 1, 10],
    "svm__loss": ["hinge", "squared_hinge"],

    # optional but often helpful: tune the estimator inside RFE
    "rfe__estimator__C": [0.01, 0.1, 1, 10],
    "rfe__estimator__loss": ["hinge", "squared_hinge"]
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV (RFE26 -> LinearSVM) on TRAIN ONLY...")
gs.fit(X_train, y_train)

print("\nBest parameters:", gs.best_params_)
print("Best mean CV F1:", gs.best_score_)

# Save grid results
out_csv = PROJECT_DIR / "q3_rfe26_linear_svm_gridsearch_results.csv"
pd.DataFrame(gs.cv_results_).to_csv(out_csv, index=False)
print("Saved full grid results to:", out_csv)

# =====================
# Evaluate best estimator on TEST
# =====================
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTEST Accuracy (RFE26 + tuned LinearSVM): {acc:.4f}\n")
print("Classification Report (TEST):\n")
print(classification_report(y_test, y_pred))

# Confusion matrix plot
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion Matrix — RFE(26) + Tuned Linear SVM (TEST)")
plt.tight_layout()

cm_path = FIG_DIR / "q3_confusion_matrix_rfe26_tuned_test.png"
plt.savefig(cm_path, dpi=200)
print("Saved:", cm_path)
plt.show()
