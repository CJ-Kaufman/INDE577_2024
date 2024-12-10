# Decision Trees and Regression Trees Implementation

## Description

This repository contains Python code for implementing decision trees and regression trees using the `sklearn` library. The two main models implemented are:

1. **Decision Trees for Classification**: The decision tree model is used to classify wine data into two categories based on selected features. The model splits the data recursively to build a tree of decision rules, and the results are visualized for better interpretation.

2. **Regression Trees for Predicting Disease Progression**: A regression tree is implemented using the diabetes dataset to predict disease progression over the course of one year based on various health metrics. The model is evaluated using mean squared error (MSE) to assess the performance of different hyperparameters.

## Datasets

### 1. Wine Dataset for Classification
The **Wine dataset** from the `sklearn` library is used for classification. It contains data about different types of wine, with features such as alcohol content, phenols, and color intensity. The dataset has been filtered to only include two classes (target values 0 and 1), and the goal is to classify the wine into these two categories.

- **Features**: Alcohol content, phenols, color intensity, etc.
- **Target**: Wine class (0 or 1).
  
### 2. Diabetes Dataset for Regression
The **Diabetes dataset** from the `sklearn` library is used for regression. It includes information on 442 patients with diabetes, and the task is to predict the progression of the disease based on several medical features (e.g., BMI, age, blood pressure).

- **Features**: Age, sex, body mass index (BMI), blood pressure, cholesterol levels, etc.
- **Target**: Disease progression after one year.

## Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/decision-trees-regression.git
cd decision-trees-regression
```

### 2. Install Dependencies

Ensure that you have the required libraries installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Running the Code

You can run the notebook in Google Colab or Jupyter Notebook:

- Open the notebook and execute the code cells step by step.
- The code will load the datasets, train the decision trees, and evaluate their performance using accuracy metrics for classification and MSE for regression.

#### Example for Classification:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load wine dataset
wine = load_wine()
X = wine.data[:, 2:4]
y = wine.target
X = X[y != 2]
y = y[y != 2]

# Train the Decision Tree
clf = DecisionTreeClassifier(max_depth=15)
clf.fit(X, y)

# Evaluate the performance and visualize the tree
print(clf.score(X, y))
```

#### Example for Regression:

```python
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)

# Train the Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```
