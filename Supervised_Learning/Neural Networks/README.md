# README for Neural Network Implementation

## 1. Description of the Algorithms Implemented

This repository contains an implementation of a neural network trained to recognize handwritten digits from the MNIST dataset. The model is designed with the following components:

- **Feedforward Neural Network**: The network has three layers, including two hidden layers and an output layer. Each hidden layer uses the ReLU (Rectified Linear Unit) activation function, while the output layer applies the softmax function to convert the output into probabilities.
- **Stochastic Gradient Descent (SGD)**: The network is trained using stochastic gradient descent, which updates the weights and biases to minimize the mean squared error (MSE) between predicted and actual labels.
- **Data Preprocessing**: The input images are flattened into column vectors and normalized. The labels are one-hot encoded to allow multi-class classification.
- **Error Calculation**: The Mean Squared Error (MSE) is used to track the error during training.

## 2. Summary of the Dataset

The dataset used for this implementation is the **MNIST dataset**, which consists of images of handwritten digits (0-9). The dataset is divided into:
- **Training set**: 60,000 images and their corresponding labels.
- **Test set**: 10,000 images and labels used to evaluate the trained model.

Each image is a 28x28 pixel grayscale image, which is flattened into a 784-dimensional vector to serve as input to the neural network.

## 3. Instructions for Reproducing Results

### Requirements:
- Python 3.x
- TensorFlow, NumPy, Matplotlib

### Step 1: Install Libraries
Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib
```

### Step 2: Load and Preprocess Data
Load the MNIST dataset and preprocess the images and labels:

```python
from tensorflow import keras
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0
```

### Step 3: Initialize and Train the Model
Create the neural network and train it:

```python
net = DenseNetwork(layers=[784, 120, 145, 120, 10])
net.train(flat_train_X, onehot_train_y)
```

### Step 4: Evaluate the Model
Check the accuracy on the test set:

```python
accuracy = sum([int(net.predict(x) == y) for x, y in zip(flat_test_X, test_y)]) / len(onehot_test_y)
print(f"Accuracy: {accuracy * 100:.2f}%")
```


