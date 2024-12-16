---

# Ensemble Methods / Random Forests Implementation

## a. Brief Description of the Algorithms Implemented

This repository implements three ensemble learning methods to classify penguin species based on physical features. The models evaluated are:

1. **Hard Voting Classifier**: Combines predictions from multiple models (Logistic Regression, Random Forest, and Support Vector Machine) using majority voting.
2. **Bagging (Bootstrap Aggregating)** : Uses multiple decision trees to reduce variance and improve classification accuracy, particularly with high bias models.
3. **Random Forests** : An extension of Bagging that introduces random feature selection to further reduce variance and prevent overfitting, resulting in a more robust model.

Each method is applied to classify penguin species from the `seaborn` dataset, with a focus on accuracy and comparison between models.

## b. Summary of the Dataset(s) Used for Each Algorithm

The dataset used for this implementation is the **Penguins dataset** from the `seaborn` library. The dataset includes features such as:
- Bill length (`bill_length_mm`)
- Bill depth (`bill_depth_mm`)
- Flipper length (`flipper_length_mm`)
- Body mass (`body_mass_g`)

Initially, the dataset includes three species of penguins (Adelie, Chinstrap, and Gentoo), but only two species (Gentoo and Chinstrap) are used for the classification tasks. The dataset is preprocessed by removing missing values and selecting two or four features for training the models.

## c. Instructions for Reproducing Your Results

1. Clone the repository or open this notebook in **Google Colab**.
2. Install the required libraries (if not already installed):
   ```bash
   pip install seaborn matplotlib numpy pandas scikit-learn mlxtend
   ```
3. Load the dataset using `seaborn.load_dataset("penguins")` and preprocess it by dropping missing values and selecting the relevant features.
4. Follow the code to train and evaluate the models:
   - Hard Voting Classifier
   - Bagging Classifier
   - Random Forest Classifier
5. The results, including accuracy scores and classification reports, will be displayed.

You can modify the dataset, models, or evaluation metrics to further explore different aspects of ensemble learning.

---
