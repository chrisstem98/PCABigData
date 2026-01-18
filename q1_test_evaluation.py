from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================
# Paths & data
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")
y_train = np.loadtxt(BASE_DIR / "train" / "y_train.txt", dtype=int)

X_test = np.loadtxt(BASE_DIR / "test" / "X_test.txt")
y_test = np.loadtxt(BASE_DIR / "test" / "y_test.txt", dtype=int)

# Activity labels
activity_labels = pd.read_csv(
    BASE_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)
id_to_activity = dict(zip(activity_labels.id, activity_labels.activity))

# =====================
# Final model (BEST from Grid Search)
# =====================
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=0.1,
        loss="squared_hinge",
        max_iter=40000,
        random_state=42
    ))
])

# =====================
# Train on FULL training set
# =====================
final_model.fit(X_train, y_train)

# =====================
# Test evaluation (ONCE)
# =====================
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nTEST Accuracy: {acc:.4f}\n")

print("Classification Report (TEST):\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=[id_to_activity[i] for i in sorted(id_to_activity)],
    digits=4
))

# =====================
# Confusion Matrix (Diagram)
# =====================
cm = confusion_matrix(y_test, y_pred, labels=sorted(id_to_activity))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[id_to_activity[i] for i in sorted(id_to_activity)]
)

plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix – Linear SVM (Test Set)")
plt.tight_layout()

# Save figure for report
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
fig_path = FIG_DIR / "q1_confusion_matrix_test.png"
plt.savefig(fig_path, dpi=200)
print("Saved confusion matrix figure to:", fig_path)

plt.show()
