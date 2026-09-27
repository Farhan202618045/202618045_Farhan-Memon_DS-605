# DS605 Lab 5 - Machine Learning with Scikit-learn and From Scratch

## Project Overview

This project is part of **DS605: Fundamentals of Machine Learning, Lab Assignment 5**.

The objective is to build both regression and classification models using the **UCI Productivity Prediction of Garment Employees** dataset, reproduce the workflow manually using **NumPy and Pandas**, and compare predictive performance and execution time. The assignment also requires an optimized manual implementation and a final comparison of the results.

## Dataset

**Dataset:** UCI Productivity Prediction of Garment Employees

The dataset contains 1,197 records and 15 original columns related to garment production, including department, team, targeted productivity, SMV, work in progress (WIP), overtime, incentives, idle time, number of workers, and actual productivity.

### Original columns

- `date`
- `quarter`
- `department`
- `day`
- `team`
- `targeted_productivity`
- `smv`
- `wip`
- `over_time`
- `incentive`
- `idle_time`
- `idle_men`
- `no_of_style_change`
- `no_of_workers`
- `actual_productivity`

## Project Tasks

### 1. Regression

The regression task predicts:

`actual_productivity`

A **Linear Regression** model is implemented using:

- Scikit-learn
- NumPy/Pandas from scratch using the closed-form solution

Regression metrics:

- MAE
- RMSE
- R²

### 2. Classification

A new target variable called `MeetsTarget` is created using the required rule:

- `MeetsTarget = 1` when `actual_productivity >= targeted_productivity`
- `MeetsTarget = 0` otherwise

A **Logistic Regression** model is implemented using:

- Scikit-learn
- NumPy from scratch using gradient descent

`actual_productivity` is not used as an input feature for classification because it is directly involved in creating `MeetsTarget`.

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score

## Data Preprocessing

The workflow uses the same fixed train-test split throughout the project:

- Training rows: **957**
- Testing rows: **240**
- Split: **80/20**
- Random seed: **42**

### Cleaning and preprocessing

1. Removed unnecessary whitespace from categorical text values.
2. Converted `date` into a datetime value.
3. Extracted `month` and `day_of_month` from `date`.
4. The `wip` column contained **506 missing values**. Missing WIP values were handled using the **training-data median**.
5. Categorical features were one-hot encoded.
6. Numerical features were standardized using training-set mean and standard deviation.

The final preprocessed feature matrix contained **36 features**:

- 11 numerical features
- 25 one-hot encoded categorical features

## Feature Groups

### Categorical features

- `quarter`
- `department`
- `day`
- `team`

### Numerical features

- `targeted_productivity`
- `smv`
- `wip`
- `over_time`
- `incentive`
- `idle_time`
- `idle_men`
- `no_of_style_change`
- `no_of_workers`
- `month`
- `day_of_month`

## Scikit-learn Results

### Linear Regression

| Metric | Result |
|---|---:|
| MAE | 0.100910 |
| RMSE | 0.140402 |
| R² | 0.360351 |
| Training Time | 0.032604 s* |
| Prediction Time | 0.003373 s* |

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 0.754167 |
| Precision | 0.777228 |
| Recall | 0.918129 |
| F1-score | 0.841823 |
| Training Time | 0.129006 s* |
| Prediction Time | 0.003499 s* |

## From-Scratch Results

### Manual Linear Regression

The Linear Regression model was implemented using the closed-form solution with NumPy matrix operations.

| Metric | Result |
|---|---:|
| MAE | 0.100910 |
| RMSE | 0.140402 |
| R² | 0.360351 |
| Training Time | 0.058825 s* |
| Prediction Time | 0.001439 s* |

The manual predictions matched the Scikit-learn predictions to floating-point precision, resulting in essentially identical MAE, RMSE, and R² values.

### Manual Logistic Regression - Baseline

The Logistic Regression model was implemented manually using:

- Sigmoid function
- Gradient descent
- Probability prediction
- Thresholding at 0.5
- NumPy vectorized operations

| Metric | Result |
|---|---:|
| Accuracy | 0.758333 |
| Precision | 0.781095 |
| Recall | 0.918129 |
| F1-score | 0.844086 |
| Training Time | 1.778590 s* |
| Prediction Time | 0.002136 s* |

## Optimized Manual Logistic Regression

The manual Logistic Regression implementation was optimized by tuning the learning rate and number of iterations and by reducing unnecessary work inside the training loop.

Optimization settings:

- Learning rate: **0.1**
- Number of iterations: **3000**

### Optimized results

| Metric | Result |
|---|---:|
| Accuracy | 0.758333 |
| Precision | 0.783920 |
| Recall | 0.912281 |
| F1-score | 0.843243 |
| Training Time | 0.964217 s* |
| Prediction Time | 0.001763 s* |

The optimized version reduced manual Logistic Regression training time from approximately **1.779 s to 0.964 s**, while keeping the classification metrics close to the baseline.

## Comparison and Key Observations

### Regression

- The Scikit-learn and manual Linear Regression implementations produced essentially identical predictions and evaluation metrics.
- This confirms that the manually implemented closed-form solution reproduced the Linear Regression workflow correctly.
- The manual Linear Regression training time was higher than Scikit-learn in the recorded run, while its prediction time was lower in that run.

### Classification

- The Scikit-learn and manual Logistic Regression models produced similar classification results.
- The manual baseline had Accuracy = 0.7583 and F1-score = 0.8441.
- The optimized manual version had Accuracy = 0.7583 and F1-score = 0.8432.
- Training time improved from 1.7786 s to 0.9642 s after optimization.
- Scikit-learn remained faster in training in the recorded run, which is expected because its implementation uses optimized numerical routines and solver code.
- The assignment does not require the manual implementation to beat Scikit-learn in execution time.

## Why the Manual and Scikit-learn Results Are Similar

The same dataset split, feature definitions, preprocessing logic, and evaluation metrics were used for the comparison. The manual Linear Regression uses the mathematical closed-form solution, which leads to predictions that closely match the Scikit-learn implementation. The Logistic Regression results are also close, although small differences occur because the manually implemented gradient-descent optimization uses different optimization settings from the solver used internally by Scikit-learn.

## Project Workflow

```text
Raw Data
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Create MeetsTarget
   ↓
Fixed 80/20 Train-Test Split
   ↓
Preprocessing
   ├── Missing Value Handling
   ├── Categorical Encoding
   └── Feature Scaling
   ↓
Scikit-learn Models
   ├── Linear Regression
   └── Logistic Regression
   ↓
From-Scratch Models
   ├── Manual Linear Regression
   └── Manual Logistic Regression
   ↓
Evaluation
   ↓
Comparison
   ↓
Manual Logistic Regression Optimization
```

## Files

A suggested GitHub repository structure is:

```text
202618045_Lab05_DS605/
│
├── data/
│   └── garments_worker_productivity.csv
│
├── 202618045_Lab05.ipynb
│
└── README.md
```

## How to Run

1. Clone or download the repository.
2. Open the project folder in VS Code.
3. Open `202618045_Lab05.ipynb`.
4. Select a working Python/Jupyter kernel.
5. Make sure the dataset is available at:

```text
data/garments_worker_productivity.csv
```

6. Run the notebook cells from top to bottom.

## Conclusion

This project demonstrates how the same machine-learning workflow can be implemented in two ways: using Scikit-learn and manually using NumPy/Pandas. The manual Linear Regression implementation reproduced the Scikit-learn results very closely. The manual Logistic Regression also produced similar classification performance, and its training time was reduced through optimization. Overall, the project shows the difference between using a high-level machine-learning library and implementing the underlying algorithms and preprocessing steps directly.

---

*Timing values marked with `*` are execution-time measurements from the notebook run and may vary slightly between runs depending on the computer and system load.*
