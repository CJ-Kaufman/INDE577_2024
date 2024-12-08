Introduction
This repository contains an implementation of Linear Regression using a custom SingleNeuron class, designed to predict one variable (bill depth) based on another (bill length) for Adelie penguins. The model is trained using a single artificial neuron with a linear activation function, and it is optimized using stochastic gradient descent (SGD). The project includes data visualization, model training, and analysis of model performance, including the effect of learning rate on training accuracy.

Key Features:
Linear Regression Implementation using a simple neural network.
Data Cleaning: Handling missing values and preprocessing data.
Model Evaluation: Error analysis and learning rate tuning.
Visualization: Plots of regression data, training error, and learned model.
Dataset
This implementation uses the Palmer Penguins dataset, focusing specifically on the Adelie species. The dataset contains information about penguins, including various physical measurements. For this example, we use two parameters from the dataset:

Bill length (mm)
Bill depth (mm)
The dataset is preprocessed to only include data for the Adelie species, and rows with missing values are removed. The cleaned data is then used for training the linear regression model.

Data Description:
Feature: bill_length_mm (length of the penguin's bill in millimeters)
Target: bill_depth_mm (depth of the penguin's bill in millimeters)
The dataset is loaded from a CSV file (palmer_penguins.csv), which you may need to upload or place in the correct directory when using the code on your local machine or on platforms like Google Colab.
Instructions for Reproducing Results
Install dependencies: If you don't have the required libraries, install them using:

bash
Copy code
pip install numpy pandas matplotlib seaborn
Load the penguins dataset: Download or upload the palmer_penguins.csv file into your environment. You can load the dataset and visualize it with:

python
Copy code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("palmer_penguins.csv")
df = df.iloc[:151][["bill_length_mm", "bill_depth_mm"]]
sns.set_theme()
plt.scatter(df.bill_length_mm, df.bill_depth_mm)
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Adelie Penguin Bill Data")
plt.show()
Define and Train the Model: Use the SingleNeuron class to train the model:

python
Copy code
node = SingleNeuron(linear_activation)
node.train(X, y, alpha=0.0001, epochs=10)
Visualize the Model: Plot the regression line after training:

python
Copy code
domain = np.linspace(np.min(X) - .5, np.max(X) + .5, 100)
plt.plot(domain, node.predict(domain.reshape(-1, 1)))
plt.scatter(X, y, color="lightseagreen")
plt.title("Penguin Bill Data with Linear Regression Line")
plt.show()
Analyze Training Error: Visualize the MSE during training:

python
Copy code
plt.plot(range(1, len(node.errors_) + 1), node.errors_)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("MSE during Training")
plt.show()

Summary of Results
The linear regression model successfully predicts the bill depth of Adelie penguins based on their bill length, with a positive correlation between the two variables.
The Mean Squared Error (MSE) decreases over time during training, indicating the model's convergence.
Learning rate has an impact on the model's training. A higher learning rate may lead to less accurate results, and it’s important to choose an optimal value for the model.
Conclusion
This project demonstrates how to implement a simple linear regression model using a custom neural network class in Python. The model can be further extended to handle more complex datasets and relationships, and future work may include experimenting with regularization techniques and different learning algorithms for improved performance.
