from pathlib import Path
import numpy as np
import pandas as pd

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

# Load labels
y_train = np.loadtxt(BASE_DIR / "train" / "y_train.txt", dtype=int)
y_test = np.loadtxt(BASE_DIR / "test" / "y_test.txt", dtype=int)

# Load activity labels
activity_labels = pd.read_csv(
    BASE_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

id_to_activity = dict(zip(activity_labels.id, activity_labels.activity))

# Map numeric labels to names
y_train_names = pd.Series(y_train).map(id_to_activity)
y_test_names = pd.Series(y_test).map(id_to_activity)

# Count samples
train_counts = y_train_names.value_counts().sort_index()
test_counts = y_test_names.value_counts().sort_index()

samples_table = pd.DataFrame({
    "Train samples": train_counts,
    "Test samples": test_counts
}).astype(int)

print("\nNumber of samples per class:")
print(samples_table)

# Save for report
out_path = PROJECT_DIR / "q1_samples_per_class.csv"
samples_table.to_csv(out_path)
print("\nSaved:", out_path)
