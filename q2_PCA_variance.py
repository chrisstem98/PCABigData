from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================
# Load TRAIN data only (no leakage)
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")

# =====================
# PCA pipeline:
# Standardize -> PCA
# =====================
# We fit ONLY on training set
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA())
])

pipe.fit(X_train)

pca = pipe.named_steps["pca"]

explained_var_ratio = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained_var_ratio)

# =====================
# Plot: cumulative explained variance vs components
# =====================
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cum_explained) + 1), cum_explained)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("PCA cumulative explained variance (fit on TRAIN set)")
plt.grid(True)
plt.tight_layout()

out_path = FIG_DIR / "q2_cumulative_explained_variance.png"
plt.savefig(out_path, dpi=200)
print("Saved figure to:", out_path)

plt.show()

# =====================
# Print a few useful checkpoints
# =====================
def components_for_threshold(threshold: float) -> int:
    return int(np.argmax(cum_explained >= threshold) + 1)

for t in [0.80, 0.90, 0.95, 0.99]:
    k = components_for_threshold(t)
    print(f"Components needed for {int(t*100)}% variance: {k}")
