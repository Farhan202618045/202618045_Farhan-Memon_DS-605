# DS605 Lab 6 - Feature Extraction and Machine Learning with Image and Text Data

## Student
Farhan Memon

## Objective

This lab explores feature extraction from image and text data and applies traditional machine-learning classification models.

## Part A - Asphalt Crack Image Classification

Dataset:
Asphalt Crack Dataset - 400 images

Processing performed:
- Read images using OpenCV
- Inspected image dimensions and channels
- Resized images to 224 × 224
- Converted images to grayscale
- Extracted mean brightness, contrast, dark-pixel ratio, bright-pixel ratio, minimum intensity, maximum intensity, median intensity, Canny edge count, and edge density
- Created one feature row per image
- Trained Logistic Regression and Random Forest classifiers
- Evaluated accuracy, precision, recall, F1-score, confusion matrix, training time, and prediction time

### Baseline results

- Logistic Regression accuracy: 93.75%
- Random Forest accuracy: 95.00%

## Part B - Email Spam Classification

Dataset:
Email Spam Classification Dataset CSV - 5,172 emails

The supplied CSV contains 3,000 existing numerical word-count features, along with the email identifier and Prediction label.

Class distribution:
- Non-spam: 3,672
- Spam: 1,500

Traditional classifiers tested:
- Multinomial Naive Bayes
- Logistic Regression

### Results

Multinomial Naive Bayes:
- Accuracy: 94.20%
- Precision: 86.81%
- Recall: 94.33%
- F1-score: 90.42%

Logistic Regression:
- Accuracy: 98.26%
- Precision: 95.78%
- Recall: 98.33%
- F1-score: 97.04%

## Part C - Representation Improvement

Two representation changes were investigated.

### Gaussian Blur before Canny

Gaussian Blur significantly reduced detected edge density, but classification performance did not improve.

### Feature dimensionality reduction

The image representation was reduced from 9 features to 6 features by removing:
- minimum intensity
- maximum intensity
- median intensity

Random Forest maintained 95.00% accuracy using 6 features, while Logistic Regression changed from 93.75% to 92.50%.

This demonstrates the trade-off between feature dimensionality, computation, and predictive performance.

## Dataset Sources

Asphalt Crack Dataset:
https://data.mendeley.com/datasets/xnzhj3x8v4/1

Email Spam Classification Dataset:
https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

## Repository Structure

```text
202618045_Lab_6/
├── 202618045_Lab06.ipynb
├── outputs/
│   ├── image_features.csv
│   └── image_features_gaussian_blur.csv
└── README.md