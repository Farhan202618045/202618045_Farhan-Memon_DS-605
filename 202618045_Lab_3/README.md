# DS605 - Lab Assignment 3
## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Farhan Memon
**Student ID:** 202618045
**Course:** DS605 - Fundamentals of Machine Learning

## Dataset
Kaggle Hotel Booking Demand
Link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Preprocessing Choices

- **Dropped `company`** — 94.3% missing values, making imputation unreliable.
- **Dropped `reservation_status` and `reservation_status_date`** — these directly encode the booking outcome (target leakage).
- **Imputed `agent`, `country`, `children`** — retained and imputed rather than dropped (13.7%, 0.4%, and ~0.003% missing respectively).
- **Outlier handling:** Checked numerical columns via boxplots and the IQR method. Most flagged values (e.g. in `adults`, `children`, `booking_changes`) reflect genuine skewed/discrete data rather than errors and were retained. Only clearly implausible values were removed: negative or unrealistic `adr` values (>5000) and bookings with zero total guests (adults=children=babies=0). **182 rows removed** in total.
- **Two preprocessing pipelines built with `ColumnTransformer`:**
  - Pipeline A: `KNNImputer(n_neighbors=5)` + `StandardScaler` (numerical), `SimpleImputer(most_frequent)` + `OneHotEncoder` (categorical)
  - Pipeline B: same, with `MinMaxScaler` instead of `StandardScaler`
- Preprocessing fitted only on training data (via `Pipeline`/`ColumnTransformer.fit_transform` on `X_train`, then `.transform` on `X_test`) to avoid data leakage.

## Models Trained
- Logistic Regression (`max_iter=1000`)
- Decision Tree Classifier (`random_state=42`)
Each trained with both Pipeline A and Pipeline B → 4 total model-pipeline combinations.

## Results Summary

| Model-Pipeline | Train Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogReg + Pipeline A (Standard) | 0.818 | 0.816 | 0.810 | 0.659 | 0.727 |
| LogReg + Pipeline B (MinMax) | 0.815 | 0.813 | 0.808 | 0.652 | 0.721 |
| DecisionTree + Pipeline A (Standard) | 0.996 | 0.859 | 0.805 | 0.818 | 0.812 |
| DecisionTree + Pipeline B (MinMax) | 0.996 | 0.859 | 0.805 | 0.819 | 0.812 |

Full table: `comparison_table.csv`
Confusion matrices: `confusion_matrices.png`

## Final Observations

1. **Best overall combination:** Decision Tree gives the highest test accuracy (~85.9%) but shows clear overfitting (99.6% train vs 85.9% test accuracy). Logistic Regression is lower (~81.6%) but generalizes far more reliably (train/test gap of only ~0.2%).
2. **Scaling effect on Logistic Regression:** StandardScaler slightly outperforms MinMaxScaler (81.6% vs 81.3% test accuracy) — consistent with theory, since gradient-based optimizers tend to work slightly better with centered, unit-variance data.
3. **Scaling effect on Decision Tree:** Negligible — test accuracy differs only in the third decimal place (85.9324% vs 85.9282%), since trees split on per-feature thresholds regardless of scale.
4. **Overfitting:** Decision Tree (no depth limit) essentially memorizes training data (99.6% train accuracy) but loses substantial performance on test data — a sign it should be regularized (e.g. `max_depth` tuning) for fairer, more generalizable results.
5. **Precision/Recall trade-off:** Logistic Regression is more conservative (higher precision ~81%, lower recall ~66%), while Decision Tree catches more actual cancellations (recall ~82%) at similar precision — a meaningful trade-off depending on business use case.

## Files in this Repository
- `202618045_Lab_3.ipynb` — full notebook (cleaning, both pipelines, both models, evaluation)
- `cleaned_hotel_bookings.csv` — cleaned base dataset used for modeling
- `comparison_table.csv` — final 4-combination comparison table
- `confusion_matrices.png` — confusion matrices for best LogReg and best Decision Tree
- `README.md` — this file