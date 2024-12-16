# INDE577_2024
Final project for INDE577 class, Dr. Davila, 2024

Note: This project was compiled with the help of Dr. Davila's lecture documents and ChatGPT

# Machine Learning Algorithms - Supervised and Unsupervised Learning

This repository contains implementations of various machine learning algorithms, organized into two main categories: **Supervised Learning** and **Unsupervised Learning**. The project is a culmination of machine learning concepts learned throughout the semester, and it aims to demonstrate both classification and regression tasks using a variety of datasets.

---

## Table of Contents

- [Supervised Learning](#supervised-learning)
  - [The Perceptron](#the-perceptron)
  - [Linear Regression](#linear-regression)
  - [Logistic Regression](#logistic-regression)
  - [Neural Networks](#neural-networks)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Decision Trees / Regression Trees](#decision-trees--regression-trees)
  - [Random Forests](#random-forests)
  - [Other Ensemble Methods (Boosting)](#other-ensemble-methods-boosting)
- [Unsupervised Learning](#unsupervised-learning)
  - [K-Means Clustering](#k-means-clustering)
  - [DBSCAN](#dbscan)
  - [Principal Component Analysis](#principal-component-analysis)
  - [Image Compression with Singular Value Decomposition (SVD)](#image-compression-with-singular-value-decomposition-svd)
- [Datasets](#datasets)
- [Reproducing the Results](#reproducing-the-results)

---

## Supervised Learning

This section contains various algorithms for both regression and classification tasks.

### The Perceptron

- **Description**: A basic linear classifier used for binary classification tasks. It learns by iteratively adjusting weights based on the mistakes made during predictions.
- **Task**: Binary classification on datasets like the Penguins dataset.

### Linear Regression

- **Description**: A statistical method to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.
- **Task**: Regression problem using datasets like Diabetes.

### Logistic Regression

- **Description**: A method for binary classification that models the probability of a binary outcome based on one or more predictor variables.
- **Task**: Binary classification problem using datasets like Penguins.

### Neural Networks

- **Description**: A computational model inspired by the human brain, which consists of layers of interconnected nodes (neurons). It's capable of learning complex patterns in both regression and classification tasks.
- **Task**: Classification and regression tasks.

### K-Nearest Neighbors (KNN)

- **Description**: A simple, instance-based learning algorithm that makes predictions based on the k closest training examples in the feature space.
- **Task**: Classification and regression problems.

### Decision Trees / Regression Trees

- **Description**: A non-linear algorithm that splits the data into branches based on feature values, making predictions at the leaves.
- **Task**: Both classification and regression tasks.

### Random Forests

- **Description**: An ensemble method based on decision trees that improves accuracy by averaging the predictions of multiple trees.
- **Task**: Classification and regression.

### Other Ensemble Methods (Boosting)

- **Description**: Includes methods like AdaBoost and Gradient Boosting, which combine weak learners to form a strong model by adjusting for misclassified examples.
- **Task**: Classification and regression tasks.

---

## Unsupervised Learning

This section contains implementations for clustering and dimensionality reduction.

### K-Means Clustering

- **Description**: A popular clustering algorithm that partitions data into k clusters based on feature similarity.
- **Task**: Unsupervised classification/clustering on various datasets.

### DBSCAN

- **Description**: A density-based clustering algorithm that groups together closely packed points and marks points in low-density regions as outliers.
- **Task**: Clustering tasks with noise handling.

### Principal Component Analysis (PCA)

- **Description**: A technique used to reduce the dimensionality of data while retaining as much variance as possible. It’s often used as a preprocessing step before applying other algorithms.
- **Task**: Dimensionality reduction for visualizing high-dimensional data.

### Image Compression with Singular Value Decomposition (SVD)

- **Description**: SVD is a matrix factorization method that can be used for dimensionality reduction in image compression, preserving the most significant features while reducing the size of the data.
- **Task**: Image compression and visualization.

---

## Datasets

The following datasets are used in this project:

- **Penguins Dataset**: A dataset containing features like bill length, bill depth, flipper length, and body mass for penguins, used for classification tasks.
- **Iris Dataset**: A well-known dataset for classification tasks, containing features of Iris flowers categorized into different species.
- **Diabetes Dataset**: A dataset used for regression tasks, predicting disease progression based on various health metrics.
- **Randomly Generated Datasets**: Used for testing and experimenting with different algorithms.

---

## Reproducing the Results

To run the algorithms and reproduce the results, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
