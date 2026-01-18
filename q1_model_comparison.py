from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================
# Paths & data
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")
y_train = np.loadtxt(BASE_DIR / "train" / "y_train.txt", dtype=int)

# Output folder for figures
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# =====================
# CV setup
# =====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================
# Models to compare
# =====================
models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),
    "LinearSVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(max_iter=30000))
    ]),
    "RBF-SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", gamma="scale"))
    ]),
    "kNN(k=5)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# =====================
# Run CV and store fold scores (for plots)
# =====================
rows = []
per_model_scores = {}  # model_name -> array of fold scores

print("Running 5-fold CV on TRAIN set...\n")

for name, pipe in models.items():
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )
    per_model_scores[name] = scores

    rows.append({
        "Model": name,
        "Mean F1 (CV)": scores.mean(),
        "Std F1": scores.std()
    })

    print(f"{name}: F1_macro = {scores.mean():.4f} ± {scores.std():.4f}")

results_df = pd.DataFrame(rows).sort_values("Mean F1 (CV)", ascending=False)
print("\nSummary:")
print(results_df)

# Save results
csv_path = PROJECT_DIR / "q1_model_comparison_cv.csv"
results_df.to_csv(csv_path, index=False)
print("\nSaved:", csv_path)

# =====================
# Plot 1: Bar chart with error bars
# =====================
order = results_df["Model"].tolist()
means = results_df["Mean F1 (CV)"].tolist()
stds = results_df["Std F1"].tolist()

plt.figure(figsize=(10, 5))
plt.bar(order, means, yerr=stds, capsize=5)
plt.ylabel("F1-macro (5-fold CV on train)")
plt.title("Model comparison (mean ± std) — Train CV")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

bar_path = FIG_DIR / "q1_cv_f1_bar.png"
plt.savefig(bar_path, dpi=200)
print("Saved:", bar_path)
plt.show()

# =====================
# Plot 2: Boxplot of fold scores per model
# =====================
data_for_boxplot = [per_model_scores[m] for m in order]

plt.figure(figsize=(10, 5))
plt.boxplot(data_for_boxplot, labels=order)
plt.ylabel("F1-macro (fold scores)")
plt.title("Model stability across CV folds — Train CV")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

box_path = FIG_DIR / "q1_cv_f1_boxplot.png"
plt.savefig(box_path, dpi=200)
print("Saved:", box_path)
plt.show()
