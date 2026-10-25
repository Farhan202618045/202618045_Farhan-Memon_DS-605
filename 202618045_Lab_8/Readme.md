
# DS605 Lab 8: Decision Trees and Support Vector Machines

## 1. Project Overview

This assignment implements Decision Trees and Linear Support Vector Machines (SVMs) from scratch, compares them with scikit-learn, and investigates nonlinear SVM kernels.

The project uses the UCI Banknote Authentication dataset for classification and a synthetic two-moons dataset for nonlinear kernel experiments.

## 2. Dataset

The Banknote Authentication dataset contains 1,372 samples, four numerical features, and a binary target.

Input features:
- variance
- skewness
- curtosis
- entropy

Target:
- class 0
- class 1

The dataset contains 762 samples of class 0 and 610 samples of class 1.

The data is split into 80% training and 20% testing using `random_state=42` and stratification. All five final models use the same split.

Two features, variance and skewness, are used for 2D decision-boundary visualizations. All four features are used for the final model comparison.

## 3. Implementation

### Part A — Decision Tree from Scratch

Implemented using NumPy:
- Gini impurity and entropy
- Feature-threshold search
- Weighted child impurity and impurity reduction
- Recursive tree construction
- Stopping conditions and majority-class prediction
- Human-readable decision rules

The manual implementation is compared with scikit-learn's `DecisionTreeClassifier`. Criterion and maximum-depth settings are explored.

### Part B — Linear SVM from Scratch

Implemented using NumPy:
- Training-set feature standardization
- Binary labels converted to -1 and +1
- Linear soft-margin SVM
- Hinge loss and subgradient descent
- Decision boundary, margin and margin-violation visualizations

The manual model is compared with scikit-learn's linear SVM. Experiments explore regularization C, learning rate, and iteration count.

### Part C — Nonlinear Kernel SVM

A synthetic two-moons dataset is generated using scikit-learn's `make_moons`.

Three SVM kernels are compared:
- Linear
- Polynomial
- Radial Basis Function (RBF)

Decision-boundary plots and classification metrics are provided.

### Part D — Final Model Comparison

Five models are evaluated on the original Banknote dataset using all four input features:
1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

The comparison includes training accuracy, test accuracy, precision, recall, F1-score, training time, and prediction time.

## 4. Final Model Results

Results from the fixed 275-sample test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Manual Decision Tree | 98.55% | 97.58% | 99.18% | 98.37% |
| Scikit-learn Decision Tree | 98.55% | 97.58% | 99.18% | 98.37% |
| Manual Linear SVM | 97.09% | 93.85% | 100.00% | 96.83% |
| Scikit-learn Linear SVM | 97.45% | 94.57% | 100.00% | 97.21% |
| Scikit-learn RBF SVM | 100.00% | 100.00% | 100.00% | 100.00% |

These are results for the particular train-test split and model settings used in this experiment.

## 5. Observations

- The manual Decision Tree achieved the same test metrics as scikit-learn's Decision Tree in the final comparison.
- Scikit-learn implementations trained substantially faster than the manual implementations in the measured runs.
- Feature scaling was applied to SVM inputs using training-set statistics only.
- Increasing Decision Tree depth from 2 to 5 improved test accuracy in the two-feature experiment.
- The RBF SVM outperformed the linear and polynomial kernels on the two-moons dataset in this run.
- The RBF SVM achieved 100% test accuracy on the original Banknote dataset for this particular split. This does not guarantee perfect performance on new data.

Hyperparameter experiments are exploratory. A separate validation set or cross-validation is preferable for selecting hyperparameters before final test evaluation.

## 6. Project Structure

```text
202618045_Lab_8/
├── data/
│   └── data_banknote_authentication.txt
├── plots/
│   ├── manual_decision_tree.png
│   ├── manual_tree_decision_boundary.png
│   ├── decision_tree_depth_comparison.png
│   ├── manual_svm_training_objective.png
│   ├── manual_svm_boundaries_margins.png
│   ├── sklearn_svm_boundaries_margins.png
│   ├── svm_C_experiment.png
│   ├── svm_learning_rate_experiment.png
│   ├── svm_iterations_experiment.png
│   ├── two_moons_dataset.png
│   ├── kernel_svm_decision_boundaries.png
│   └── final_model_accuracy_comparison.png
├── results/
│   ├── decision_tree_depth_criterion_comparison.csv
│   ├── svm_hyperparameter_experiments.csv
│   ├── kernel_svm_comparison.csv
│   └── final_model_comparison.csv
├── 202618045_Lab08.ipynb
└── README.md
```

## 7. Requirements and Execution

Install the required libraries:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

Open `202618045_Lab08.ipynb` in VS Code and run the notebook cells in order.

The dataset is read from the `data/` directory, and visualizations and result tables are saved to `plots/` and `results/`.

## 8. Conclusion

This assignment demonstrates how Decision Trees and Linear SVMs can be implemented from scratch, how their behavior compares with scikit-learn, and how kernel functions allow SVMs to model nonlinear boundaries.
