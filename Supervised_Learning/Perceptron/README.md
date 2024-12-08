This repository contains an implementation of the **Perceptron algorithm** using gradient descent for binary classification.

### Algorithm:
- **Perceptron with Gradient Descent**: The Perceptron is a linear classifier that updates weights iteratively based on misclassified samples using the gradient descent algorithm. It finds an optimal decision boundary for classification.

### Dataset:
- **Palmer Penguins Dataset**: This dataset contains data about three species of penguins (Adelie, Chinstrap, and Gentoo) with attributes such as bill length, bill depth, flipper length, and body mass. The goal is to classify penguins based on these features. This dataset includes some missing values (NaN rows).

### Reproducing Results:
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the `main.py` file to execute the Perceptron algorithm on the Palmer Penguins dataset:
   ```bash
   python main.py
   ```
This will train the model and output the accuracy of the classifier.
