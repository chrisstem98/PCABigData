# PCAProject - Repository Description

This repository implements classification experiments on the **UCI HAR Dataset**, with a primary focus on:
- baseline classification,
- feature selection/reduction (PCA, RFE),
- evaluation with proper train/test separation (no leakage).

The description below is based **only** on the `.py` files in the `PCABigData` folder.

## Data Structure Used
The scripts read data from:
- `PCABigData/data/UCI HAR Dataset/train/X_train.txt`
- `PCABigData/data/UCI HAR Dataset/train/y_train.txt`
- `PCABigData/data/UCI HAR Dataset/test/X_test.txt`
- `PCABigData/data/UCI HAR Dataset/test/y_test.txt`
- `PCABigData/data/UCI HAR Dataset/activity_labels.txt`
- `PCABigData/data/UCI HAR Dataset/features.txt`

## What Each Script Does

### 1) `q1_load_check.py`
- Loads train/test matrices and labels.
- Prints shapes and unique classes.
- Runs assertions to verify 561 features and labels 1..6.

### 2) `q1_samples_per_class.py`
- Maps numeric labels to activity names.
- Computes samples per class for train/test.
- Saves the table to `q1_samples_per_class.csv`.

### 3) `q1_model_comparison.py`
- Compares 5 models with 5-fold Stratified CV using `f1_macro`:
  - Logistic Regression
  - Linear SVM
  - RBF SVM
  - kNN (k=5)
  - Random Forest
- Saves summary results to `q1_model_comparison_cv.csv`.
- Produces 2 figures in `figures/`:
  - `q1_cv_f1_bar.png`
  - `q1_cv_f1_boxplot.png`

### 4) `q1_linear_SVM_gridsearch.py`
- Defines pipeline: `StandardScaler -> LinearSVC`.
- Runs `GridSearchCV` (5-fold stratified, `f1_macro`) on the train set only.
- Tunes:
  - `svm__C`: [0.01, 0.1, 1, 10, 100]
  - `svm__loss`: [`hinge`, `squared_hinge`]
- Saves full CV results to `q1_linear_svm_gridsearch_results.csv`.

### 5) `q1_test_evaluation.py`
- Trains final pipeline `StandardScaler -> LinearSVC` with fixed hyperparameters:
  - `C=0.1`, `loss='squared_hinge'`, `max_iter=40000`
- Fits on the full train set and evaluates once on the test set.
- Prints test accuracy and classification report.
- Saves confusion matrix figure to `figures/q1_confusion_matrix_test.png`.

### 6) `q2_PCA_variance.py`
- Fits `StandardScaler -> PCA()` on the train set only.
- Computes cumulative explained variance.
- Prints how many components are needed for 80%, 90%, 95%, 99% variance.
- Saves plot to `figures/q2_cumulative_explained_variance.png`.

### 7) `q3_pca80_linear_svm_gridsearch_test.py`
- Defines pipeline `StandardScaler -> PCA(n_components=26) -> LinearSVC`.
- Runs `GridSearchCV` on the train set only for `svm__C` and `svm__loss`.
- Saves results to `q3_pca80_linear_svm_gridsearch_results.csv`.
- Evaluates best estimator on the test set.
- Saves confusion matrix to `figures/q3_confusion_matrix_pca80_tuned_test.png`.

### 8) `q3_rfe26_linear_svm_gridsearch_test.py`
- Defines pipeline `StandardScaler -> RFE(n_features_to_select=26) -> LinearSVC`.
- RFE uses an internal `LinearSVC` estimator for feature ranking.
- Runs `GridSearchCV` on the train set for:
  - final classifier (`svm__C`, `svm__loss`)
  - estimator inside RFE (`rfe__estimator__C`, `rfe__estimator__loss`)
- Saves results to `q3_rfe26_linear_svm_gridsearch_results.csv`.
- Evaluates on the test set and saves confusion matrix to `figures/q3_confusion_matrix_rfe26_tuned_test.png`.

### 9) `q4_pca_feature_importance.py`
- Applies standardization and PCA with `n_components=26` on the train set.
- Computes PCA-based importance per original feature as:
  - sum of `(loading^2 * explained_variance_ratio)` across 26 PCs,
  - then normalization so importances sum to 1.
- Merges with feature names from `features.txt`.
- Keeps the top 26 most important features (`head(26)`).
- Saves table to `q4_pca_top10_feature_importance.csv` (filename says top10, but code keeps 26).
- Saves plot to `figures/q4_pca_top26_feature_importance.png`.

## Outputs Produced by the Code
- CSV files with grid search/model comparison/sample count results.
- Figures in `PCABigData/figures/` for performance and confusion matrices.
- Classification metric reports in terminal output.

## Methodology Note
In all scripts with tuning/feature reduction, model selection is fitted on the training set with cross-validation, and the test set is used for final evaluation.
