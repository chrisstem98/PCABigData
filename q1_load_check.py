from pathlib import Path
import numpy as np

# 1) Paths and directories
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# 2) Load data
X_train = np.loadtxt(TRAIN_DIR / "X_train.txt")
y_train = np.loadtxt(TRAIN_DIR / "y_train.txt", dtype=int)

X_test = np.loadtxt(TEST_DIR / "X_test.txt")
y_test = np.loadtxt(TEST_DIR / "y_test.txt", dtype=int)

# 3) Print shapes
print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# 4) Sanity checks
print("\nSanity checks:")
print("n_features (train):", X_train.shape[1])
print("Unique labels (train):", np.unique(y_train))
print("Unique labels (test) :", np.unique(y_test))

# 5) Basic assertions to verify correctness
assert X_train.shape[1] == 561, "Expected 561 features."
assert X_test.shape[1] == 561, "Expected 561 features."
assert set(np.unique(y_train)) == {1,2,3,4,5,6}, "Expected labels 1..6 in train."
assert set(np.unique(y_test)) == {1,2,3,4,5,6}, "Expected labels 1..6 in test."

print("\Step 1 OK: Data loaded correctly.")
