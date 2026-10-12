# Scikit-learn Model Benchmarking

## Overview

This project explores and compares machine learning models for two types of supervised learning tasks: regression and binary classification. It focuses on building a reproducible workflow for data preparation, model training, hyperparameter tuning, model evaluation, and performance comparison using Python and scikit-learn.

The project uses two datasets provided by scikit-learn:

1. **Diabetes Dataset** for regression.
2. **Breast Cancer Wisconsin (Diagnostic) Dataset** for binary classification.

The goal is to understand how different models perform on different prediction tasks and how evaluation metrics can help guide model selection.

## Project Objectives

* Explore and understand real-world machine learning datasets.
* Prepare data for regression and classification tasks.
* Establish baseline models for performance comparison.
* Train and compare multiple machine learning algorithms.
* Evaluate models using appropriate metrics.
* Use validation data for model selection and hyperparameter tuning.
* Evaluate the selected model on held-out test data.
* Visualize model performance and evaluation results.
* Organize the workflow into reusable Python modules.

## Datasets

### 1. Diabetes Dataset

**Task:** Regression

**Source:** [Scikit-learn Diabetes Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

The Diabetes dataset is a regression dataset included in scikit-learn. It contains information about 442 patients and 10 baseline features. The target is a quantitative measure of disease progression one year after baseline.

#### Dataset Details

| Property                 | Description                                 |
| ------------------------ | ------------------------------------------- |
| Dataset name             | Diabetes Dataset                            |
| Number of observations   | 442                                         |
| Number of input features | 10                                          |
| Learning task            | Regression                                  |
| Target variable          | Quantitative measure of disease progression |
| Data source              | `sklearn.datasets.load_diabetes()`          |

#### Input Features

The dataset contains the following 10 input features:

| Feature | Description                    |
| ------- | ------------------------------ |
| `age`   | Age                            |
| `sex`   | Sex                            |
| `bmi`   | Body mass index                |
| `bp`    | Average blood pressure         |
| `s1`    | First blood serum measurement  |
| `s2`    | Second blood serum measurement |
| `s3`    | Third blood serum measurement  |
| `s4`    | Fourth blood serum measurement |
| `s5`    | Fifth blood serum measurement  |
| `s6`    | Sixth blood serum measurement  |

The feature values in scikit-learn's default version of this dataset are mean-centered and scaled.

#### Regression Models

The project is designed to compare the following approaches:

* Baseline regression model
* Linear regression
* Decision tree regression
* Random forest regression
* XGBoost regression

#### Regression Evaluation Metrics

The models will be evaluated using:

* **Root Mean Squared Error (RMSE):** Measures prediction error, penalizing larger errors more heavily.
* **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
* **R² Score:** Measures how well the model explains variation in the target variable.

Lower RMSE and MAE values indicate smaller prediction errors. A higher R² generally indicates a better fit to the observed target values.

### 2. Breast Cancer Wisconsin (Diagnostic) Dataset

**Task:** Binary classification

**Source:** [Scikit-learn Breast Cancer Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

The Breast Cancer Wisconsin (Diagnostic) dataset is a binary classification dataset available in scikit-learn. It contains 569 observations and 30 numeric input features computed from digitized images of fine needle aspirates of breast masses.

The classification task is to distinguish between malignant and benign diagnoses.

#### Dataset Details

| Property                 | Description                             |
| ------------------------ | --------------------------------------- |
| Dataset name             | Breast Cancer Wisconsin (Diagnostic)    |
| Number of observations   | 569                                     |
| Number of input features | 30                                      |
| Number of classes        | 2                                       |
| Learning task            | Binary classification                   |
| Data source              | `sklearn.datasets.load_breast_cancer()` |

#### Input Features

The 30 features are derived from measurements of cell nuclei. They cover 10 characteristics:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each characteristic, the dataset provides three measurements: mean, standard error, and worst value. Together, these produce 30 input features.

#### Target Labels

For this project, the target labels will be interpreted as follows:

| Label | Meaning   |
| ----- | --------- |
| `1`   | Malignant |
| `0`   | Benign    |

**Note:** Scikit-learn's original target encoding is the opposite: `0` represents malignant and `1` represents benign. Any conversion to the encoding above must be applied consistently throughout training, evaluation, and visualization.

#### Classification Models

The project is designed to compare:

* Baseline classifier
* Logistic regression
* Decision tree classifier
* Random forest classifier
* XGBoost classifier

#### Classification Evaluation Metrics

The models will be evaluated using several complementary metrics:

* **Log Loss:** Evaluates the quality of predicted probabilities.
* **ROC AUC:** Measures the model's ability to rank positive examples above negative examples.
* **Accuracy:** Measures the proportion of correct predictions.
* **Precision:** Measures the proportion of predicted positive cases that are actually positive.
* **Recall (Sensitivity):** Measures the proportion of actual positive cases correctly identified.
* **Specificity:** Measures the proportion of actual negative cases correctly identified.
* **F1 Score:** Combines precision and recall into one metric.
* **Confusion Matrix:** Summarizes true positives, true negatives, false positives, and false negatives.

The project also considers probability calibration and how changing the classification threshold affects model performance.

These models and metrics are used for educational evaluation only. The results are not intended for clinical diagnosis or medical decision-making.

## Methodology

### 1. Data Loading

Load the datasets using scikit-learn's built-in dataset functions. This makes the experiments easier to reproduce without manually downloading the raw data files.

### 2. Data Preparation

Prepare the features and targets for the appropriate learning task. Check feature shapes, target labels, and data types before training.

Any preprocessing that learns parameters from the data should be fitted only on the training portion of the data.

### 3. Train, Validation, and Test Splits

Separate the data into training, validation, and test sets.

* **Training set:** Used to fit model parameters.
* **Validation set:** Used to compare models, tune hyperparameters, and select classification thresholds.
* **Test set:** Used for final evaluation after the modeling decisions have been made.

The test set should not be repeatedly used to choose models or tune parameters. Keeping it separate provides a more reliable estimate of performance on unseen data.

### 4. Baseline Models

Establish simple baseline models to provide a reference point for evaluating more complex algorithms.

A more complicated model is not necessarily better simply because it uses more parameters or has a more advanced algorithm.

### 5. Model Training and Comparison

Train the selected regression and classification models using consistent data splits wherever appropriate. Record the relevant metrics so that model performance can be compared systematically.

### 6. Hyperparameter Tuning

Use validation data or an appropriate cross-validation procedure to compare hyperparameter settings.

Keep the test set separate from this process to reduce the risk of overly optimistic evaluation.

### 7. Final Evaluation

Evaluate the selected models on the held-out test data and summarize the results using tables and visualizations.

## Technology Stack

* **Python** for implementation
* **NumPy** for numerical operations
* **pandas** for tabular data handling
* **Matplotlib** for plotting
* **scikit-learn** for datasets, preprocessing, models, and evaluation metrics
* **XGBoost** for gradient-boosted tree models
* **Git and GitHub** for version control and project hosting
* **Visual Studio Code** as the development environment


## Results and Visualizations

Model comparison tables, plots, and written findings will be added after the experiments have been implemented and run.

Potential visualizations include:

* Regression metric comparisons
* Actual versus predicted regression values
* Classification confusion matrices
* ROC curves
* Precision-recall comparisons
* Probability calibration plots
* Classification threshold comparisons

No performance numbers are included here because results should be generated from the actual implementation rather than estimated or invented.

## Limitations

* The datasets are relatively small and may not represent the full complexity of real-world prediction problems.
* Results depend on data splitting, preprocessing, model configurations, and hyperparameter choices.
* A strong score on a held-out test set does not guarantee good performance on every future dataset.
* Classification metrics may reflect different trade-offs between false positives and false negatives.
* The breast cancer dataset is provided for educational machine learning experiments and should not be used as a clinical decision-making system.

## Future Improvements

Possible extensions include:

* Adding cross-validation
* Comparing additional algorithms
* Improving automated hyperparameter search
* Adding automated tests for data preparation and evaluation functions
* Recording experiment configurations
* Improving plots and result summaries
* Adding a command-line interface for selecting experiments
* Documenting model limitations and reproducibility considerations

## References

1. Scikit-learn, [Diabetes Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).
2. Scikit-learn, [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
3. University of Wisconsin, [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
4. Scikit-learn, [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
