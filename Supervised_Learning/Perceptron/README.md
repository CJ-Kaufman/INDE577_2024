# Perceptron Implementation README

## Introduction

This repository provides an implementation of a **Perceptron model** for binary classification. The Perceptron is a simple neural network model that is capable of separating linearly separable data into two classes. In this example, the dataset used consists of measurements from two types of penguins, **Chinstrap** and **Gentoo**, from the *Palmer Penguins* dataset. The goal of this implementation is to train the Perceptron model to classify penguins into these two species based on their flipper length and bill depth measurements.

## Project Overview

The Perceptron model is trained using a set of two numerical features: **flipper length** and **bill depth**, to classify penguin species into two categories. The code includes the following steps:
1. **Data Preparation**: Loading and cleaning the Palmer Penguins dataset.
2. **Model Training**: Implementing a Perceptron model, with functions for:
   - Pre-activation and post-activation calculations
   - Loss function
   - Gradient descent optimization
3. **Model Evaluation**: Training the Perceptron and evaluating its accuracy on the dataset.

## Prerequisites

To run this code, you will need the following Python libraries:
- `numpy`: For numerical computations.
- `pandas`: For handling data manipulation.
- `matplotlib`: For plotting graphs.
- `seaborn`: For visualization and dataset loading.

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn
```

## Code Walkthrough

### 1. Imports and Data Loading
The dataset is loaded using the `seaborn` library, which provides access to the **Palmer Penguins dataset**.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# Load the Palmer Penguins dataset
penguins = sns.load_dataset("penguins")
```

The dataset contains several features like bill length, bill depth, flipper length, and body mass, along with the species (penguin types).

### 2. Data Preparation
We filter the dataset to only include the **Chinstrap** and **Gentoo** species and select two features: **flipper length** and **bill depth**.

```python
chinstrap_gentoo = penguins.query("species=='Chinstrap' or species=='Gentoo'")
measurements = chinstrap_gentoo[["species", "flipper_length_mm", "bill_depth_mm"]]
```

Next, we clean the data by removing rows containing `NaN` values and map the species to numerical values (1 for **Chinstrap** and -1 for **Gentoo**).

```python
label_mapping = {'Chinstrap': 1, 'Gentoo': -1}
y = np.array([label_mapping[species] for species in measurements['species']])

# Cleaning the data to remove NaN values
X_dirty = measurements[["flipper_length_mm", "bill_depth_mm"]].to_numpy()
mask = ~np.isnan(X_dirty).any(axis=1)
X = X_dirty[mask]
y = y[mask]
```

### 3. Defining the Perceptron Model
We initialize the **weights** and **bias** of the perceptron randomly and define the **activation function** using the sign function.

```python
w = np.random.randn(2)  # Random initial weights
b = np.random.randn()   # Random initial bias

# Pre-activation function
def preactivation(w, b, x):
    return np.dot(w, x) + b

# Post-activation function (sign function)
def post_activation(z):
    return np.sign(z)
```

The **neuron function** calculates the output of the perceptron based on the weights, bias, and input data.

```python
def neuron(w, b, x):
    return post_activation(preactivation(w, b, x))
```

### 4. Loss Function and Gradient Descent
The **loss function** measures how far the model's predictions are from the actual values. We use **mean squared error** as the loss metric.

```python
def loss(X, y, w, b):
    total_loss = 0
    for x, y_true in zip(X, y):
        y_pred = neuron(w, b, x)
        total_loss += (y_pred - y_true) ** 2
    return total_loss / len(X)
```

The **gradient descent update** function adjusts the weights and bias to minimize the loss function.

```python
def gradient_descent_update(w, b, x, y, learning_rate=0.05):
    y_hat = neuron(w, b, x)
    error = y_hat - y
    w = w - learning_rate * error * x
    b = b - learning_rate * error
    return w, b
```

The **gradient descent** function iterates through the dataset for a specified number of epochs, updating the model parameters each time.

```python
def gradient_descent(X, y, w, b, learning_rate=0.001, epochs=100):
    losses = [loss(X, y, w, b)]
    for epoch in range(epochs):
        for x, y_true in zip(X, y):
            w, b = gradient_descent_update(w, b, x, y_true, learning_rate)
        losses.append(loss(X, y, w, b))
    return w, b, losses
```

### 5. Model Evaluation
Once the model is trained, we evaluate it by predicting the species labels and comparing them to the true labels.

```python
w, b, losses = gradient_descent(X, y, w, b, epochs=1000)

# Plotting the loss function over epochs
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Predictions
y_hat = predict(X, w, b)
accuracy = np.mean(y_hat == y)
print(f"Model Accuracy: {accuracy * 100}%")
```

### 6. Results
After training, the model should classify the data points with high accuracy. A plot of the loss function will show how the loss decreases over time, demonstrating the effectiveness of the training process.

## Conclusion

This implementation demonstrates how to use a Perceptron model to classify two types of penguin species based on their physical characteristics. The model is trained using a gradient descent algorithm, and the final output is a classification accuracy near 100%. The project is a good starting point for learning about neural networks and binary classification.

## How to Use

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/perceptron-implementation.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `perceptron_model.ipynb` notebook in Google Colab or Jupyter Notebook to see the results and modify the model as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
