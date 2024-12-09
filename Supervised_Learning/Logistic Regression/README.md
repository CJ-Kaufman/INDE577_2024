# Logistic Regression Implementation

## Description

This project demonstrates the implementation of logistic regression for binary classification using the **Palmer Penguins** dataset. The dataset involves two penguin species, **Chinstrap** and **Gentoo**, with features such as body mass and flipper length. The main goal is to use logistic regression to classify the penguins based on their body mass, utilizing the sigmoid function to handle non-linearly separable data.

### Key Steps:
1. **Data Loading & Preprocessing**: 
   - The dataset is loaded and cleaned, converting physical measurements to appropriate units.
2. **Data Visualization**: 
   - Visualizing the penguin data in a scatter plot to observe patterns.
3. **Sigmoid Function Implementation**: 
   - The sigmoid function is used to map predictions between 0 and 1, representing class probabilities.
4. **Training a Logistic Regression Model**: 
   - A single neuron model is trained on the cleaned data, optimizing weights using gradient descent to minimize the cross-entropy loss.
5. **Visualization**: 
   - Training progress and decision boundaries are plotted to assess the model’s performance.

## Dataset

The dataset used in this project is the **Palmer Penguins** dataset, which contains data on three species of penguins, including:
- **Species**: Chinstrap, Gentoo (only these two are used for classification).
- **Body Mass (g)**: The weight of the penguin in grams.
- **Flipper Length (mm)**: The length of the penguin's flipper in millimeters.
  
For the purpose of this classification task, we focus on **Chinstrap** and **Gentoo** penguins, using body mass as the primary feature for classification.

## Instructions to Reproduce Results

1. **Setup**: Open the notebook in Google Colab.
2. **Data Upload**: Upload the `palmer_penguins.csv` file to your environment. The dataset can be loaded using the `pd.read_csv()` function.
3. **Run the Code**: Execute the code cells sequentially, starting from importing libraries to training the logistic regression model.
4. **Visualization**: Check the plots generated for data distribution, sigmoid function visualization, and the cost function over epochs.
5. **Training & Prediction**: After training the model, you can predict the probability of penguins belonging to the Gentoo species based on body mass.
