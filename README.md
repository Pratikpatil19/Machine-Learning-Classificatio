# Machine Learning Classification Examples

This repository contains examples of machine learning classification using scikit-learn in Python. It demonstrates how to apply the K-Nearest Neighbors (KNN) algorithm to three different datasets: breast cancer, wine, and diabetes.


## Datasets

- **Breast Cancer Wisconsin (Diagnostic) Dataset:** This dataset is used for breast cancer detection using features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Wine Dataset:** This dataset is used for wine classification based on chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.
- **Pima Indians Diabetes Dataset:** This dataset is used for diabetes detection based on various medical predictor variables.


## Code Overview

The code is organized into three sections, one for each dataset. Each section includes:

1. **Data Loading:** Loading the dataset using `sklearn.datasets`.
2. **Data Exploration:** Exploring the dataset using `pandas` and `matplotlib`.
3. **Model Training:** Training a KNN classifier using `sklearn.neighbors.KNeighborsClassifier`.
4. **Model Evaluation:** Evaluating the classifier using `sklearn.model_selection.train_test_split` and `knn.score`.


## Usage

To run the code, you need to have Python 3 installed along with the following libraries:

- scikit-learn
- pandas
- numpy
- matplotlib

You can install these libraries using pip:
