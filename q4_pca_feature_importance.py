import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# =====================
# Paths & data
# =====================
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

X_train = np.loadtxt(BASE_DIR / "train" / "X_train.txt")

# =====================
# Standardization
# =====================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# =====================
# PCA (80% variance -> 26 components)
# =====================
pca = PCA(n_components=26)
pca.fit(X_train_sc)

# =====================
# PCA-based feature importance
# =====================
loadings = pca.components_                    # (26, 561)
var_ratio = pca.explained_variance_ratio_     # (26,)

# Σ (loading^2 * explained_variance_ratio)
feature_importance = np.sum(
    (loadings ** 2) * var_ratio[:, np.newaxis],
    axis=0
)

# Normalize for readability
feature_importance /= feature_importance.sum()

# =====================
# Create DataFrame (indices)
# =====================
importance_df = pd.DataFrame({
    "feature_index": np.arange(X_train.shape[1]),
    "pca_importance": feature_importance
}).sort_values("pca_importance", ascending=False)

# =====================
# Load feature names
# =====================
features_df = pd.read_csv(
    BASE_DIR / "features.txt",
    sep=r"\s+",
    header=None,
    names=["feature_index", "feature_name"]
)

# indices in features.txt start from 1
features_df["feature_index"] -= 1

# =====================
# Merge importance with feature names
# =====================
importance_named_df = importance_df.merge(
    features_df,
    on="feature_index",
    how="left"
)

# =====================
# Keep Top-10 features
# =====================
top10_df = importance_named_df.head(26)

# Save table for Word
out_csv = PROJECT_DIR / "q4_pca_top10_feature_importance.csv"
top10_df.to_csv(out_csv, index=False)
print("Saved table:", out_csv)

print("\nTop-10 PCA-based features:")
print(top10_df)

# =====================
# Plot Top-10 feature importance
# =====================
plt.figure(figsize=(8, 5))
plt.barh(
    top10_df["feature_name"][::-1],
    top10_df["pca_importance"][::-1]
)
plt.xlabel("PCA-based feature importance")
plt.title("Top-26 Features Based on PCA Loadings")
plt.tight_layout()

fig_path = FIG_DIR / "q4_pca_top26_feature_importance.png"
plt.savefig(fig_path, dpi=200)
print("Saved figure:", fig_path)
plt.show()
