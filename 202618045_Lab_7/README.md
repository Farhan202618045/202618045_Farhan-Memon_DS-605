# Image Representation, PCA, Eigenfaces and Face Recognition

**Course:** DS605 — Fundamentals of Machine Learning  
**Assignment:** Lab Assignment 7  
**Dataset:** AT&T Database of Faces (ORL)

## 1. Project Overview

This project explores image representation, dimensionality reduction using Principal Component Analysis (PCA), eigenfaces, image reconstruction, and face recognition.

The AT&T Database of Faces contains 400 grayscale images of 40 subjects, with 10 images per subject. Each image has dimensions of 112 × 92 pixels.

The project implements PCA using NumPy linear-algebra operations, analyzes explained variance, reconstructs faces from reduced representations, and compares similarity-based face recognition with traditional machine-learning classifiers.

## 2. Objectives

- Convert grayscale images into numerical pixel feature vectors.
- Build the image matrix with corresponding subject labels.
- Create a reproducible train-test split in which every subject appears in both sets.
- Implement PCA using eigen-decomposition and Singular Value Decomposition (SVD) with NumPy.
- Calculate explained variance and select the minimum number of components needed to preserve at least 90% variance.
- Visualize eigenfaces and reconstruct sample faces.
- Compare raw-pixel and PCA-based face recognition.
- Train and evaluate KNN and SVM classifiers on PCA features.
- Study how the number of retained components affects recognition accuracy and representation size.

## 3. Dataset and Image Representation

The dataset contains:

- 400 face images
- 40 subjects
- 10 images per subject
- Image dimensions: 112 × 92 pixels
- 10,304 pixel features per flattened image

Each grayscale image is converted into a NumPy array and flattened into a one-dimensional feature vector.

The complete image matrix has shape `(400, 10304)`, while the label array has shape `(400,)`.

A reproducible stratified split is implemented using NumPy with seed 42.

| Dataset | Images | Images per subject |
|---|---:|---:|
| Training | 320 | 8 |
| Testing | 80 | 2 |

All 40 subjects appear in both sets.

## 4. PCA and Eigenfaces

The mean face is calculated from the training images only. Both training and test images are centered using the same training mean.

Two PCA approaches are implemented using NumPy:

1. **Eigen-decomposition:** A sample-space Gram matrix is calculated from the centered training matrix. Its eigenvalues and eigenvectors are used to recover principal directions in pixel space.
2. **SVD:** NumPy's Singular Value Decomposition is applied to the centered training matrix. The right singular vectors provide the principal directions.

The two methods produce matching principal directions and eigenvalues up to numerical precision, with possible sign changes in eigenvectors.

Scikit-learn PCA is not used.

The principal directions are reshaped into 112 × 92 images to visualize the eigenfaces. Eigenfaces represent patterns of pixel variation and do not correspond directly to photographs of individual subjects.

## 5. Explained Variance and Dimensionality Reduction

The centered training data yielded 319 numerically meaningful principal directions.

The minimum number of components required to retain at least 90% of the training variance was **100 components**, preserving approximately **90.0905%**.

| Representation | Features per image |
|---|---:|
| Original image | 10,304 |
| PCA representation | 100 |

This corresponds to a **99.03% reduction in feature count**.

The cumulative explained-variance plot is saved as `outputs/cumulative_explained_variance.png`.

## 6. Image Reconstruction

Test images are projected onto the first 100 principal directions. They are reconstructed using their PCA scores, the selected principal directions, and the training mean face.

The reconstruction metrics on the 80 test images were:

| Metric | Result |
|---|---:|
| Mean Squared Error (MSE) | 314.33 |
| Root Mean Squared Error (RMSE) | 17.73 |
| Peak Signal-to-Noise Ratio (PSNR) | 23.16 dB |

The reconstructed faces retain recognizable facial structures, although some fine details are lost due to dimensionality reduction.

### Storage analysis

| Item | Size |
|---|---:|
| Original training pixel array | 3,297,280 bytes |
| Reduced training PCA scores | 256,000 bytes |
| Shared PCA components and mean face | 8,325,632 bytes |
| Reduced scores plus shared PCA model | 8,581,632 bytes |

Each original image uses 10,304 bytes of `uint8` pixel storage. Each 100-component representation uses 800 bytes of `float64` scores, excluding the shared PCA model.

Although the per-image numerical payload is reduced, storing the shared PCA components and mean face introduces overhead. Therefore, the reduced representation does not necessarily require less total storage for this small dataset.

## 7. Face Recognition Results

Nearest-face recognition uses Euclidean distance to find the closest stored training face for each test image. Raw-pixel and PCA-based recognition use the same training and test split.

| Method | Features per image | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| Raw-pixel nearest neighbor | 10,304 | 77 | 3 | 96.25% |
| PCA nearest neighbor | 100 | 77 | 3 | 96.25% |
| KNN baseline | 100 | 71 | 9 | 88.75% |
| SVM baseline | 100 | 77 | 3 | 96.25% |
| Tuned KNN | 100 | 77 | 3 | 96.25% |
| Tuned SVM | 100 | 78 | 2 | 97.50% |

The raw-pixel and PCA nearest-neighbor methods achieved the same observed test accuracy, even though PCA reduced the image representation from 10,304 features to 100.

The initial KNN classifier performed less well than the nearest-neighbor baseline. Hyperparameter tuning improved its test accuracy to 96.25%. The tuned SVM achieved the highest observed test accuracy of 97.50%, or 78 correct predictions out of 80.

These figures describe results on this particular test split and should not be interpreted as guarantees of performance on other face images.

## 8. Component-Count Experiment

The experiment evaluates 25, 50, 75, 98, 125, and 150 principal components.

| Components | Variance preserved | Nearest neighbor | KNN (3 neighbors) | SVM |
|---:|---:|---:|---:|---:|
| 25 | 73.39% | 95.00% | 90.00% | 91.25% |
| 50 | 82.20% | 96.25% | 93.75% | 96.25% |
| 75 | 86.94% | 96.25% | 91.25% | 96.25% |
| 98 | 89.88% | 96.25% | 91.25% | 97.50% |
| 125 | 92.39% | 96.25% | 77.50% | 96.25% |
| 150 | 94.18% | 96.25% | 78.75% | 93.75% |

The results show that greater explained variance does not necessarily lead to improved classification accuracy.

For example, nearest-neighbor recognition achieved 96.25% accuracy with 50 components, preserving 82.20% of the training variance. The main PCA representation nevertheless uses 100 components because that is the minimum count that satisfies the assignment's 90% variance requirement.

The component-count table is saved as `outputs/component_count_experiment.csv`.

## 9. Hyperparameter Tuning and Evaluation

KNN and SVM hyperparameters were explored using four-fold stratified cross-validation on the training data.

The tested KNN settings included the number of neighbors, neighbor weighting, distance metric, and feature scaling. SVM experiments varied the regularization parameter and RBF kernel parameter.

The selected configurations were then evaluated on the held-out test set. The best observed tuned-model result was 97.50% test accuracy for SVM.

**Evaluation limitation:** PCA was calculated on the complete training set before the exploratory cross-validation folds were created. This may make cross-validation results optimistic because validation-fold data influenced the PCA basis. A stricter evaluation would fit PCA separately within each cross-validation training fold, using a NumPy implementation, before transforming its validation fold.

The component-count experiment also evaluates several configurations on the same held-out test set. Its results are useful for exploratory comparison but should not be treated as an independent, unbiased estimate after selecting the best-performing configuration.

## 10. Output Files

The `outputs` directory contains the figures and numerical results generated by the notebook.

### Figures

- `cumulative_explained_variance.png