# Principal Component Analysis (PCA) on the Penguins Dataset

## a. Brief Description of the Algorithms Implemented

This project implements **Principal Component Analysis (PCA)**, a technique used for dimensionality reduction. PCA transforms high-dimensional data into fewer dimensions by identifying the most significant components (principal components) that retain the maximum variance in the dataset. The goal of PCA is to simplify the dataset while retaining the most important features. This implementation uses **Singular Value Decomposition (SVD)** to compute the principal components and reduces the dataset from 4 dimensions to 2 for visualization purposes.

## b. Summary of the Dataset(s) Used for Each Algorithm

The dataset used for PCA is the **Penguins dataset** from the **Seaborn library**, which contains data about penguins from three species: **Adelie**, **Chinstrap**, and **Gentoo**. The dataset includes the following features:
- **bill_length_mm**: Length of the bill (in mm)
- **bill_depth_mm**: Depth of the bill (in mm)
- **flipper_length_mm**: Length of the flipper (in mm)
- **body_mass_g**: Mass of the penguin (in grams)

The dataset has been cleaned to remove any missing (NaN) values in these features, ensuring a complete dataset for analysis.

## c. Instructions for Reproducing Results

1. **Clone the repository** or download the notebook from GitHub.
2. Open the notebook in **Google Colab** or any Python environment.
3. Install the necessary packages (if not already installed):
   ```bash
   pip install seaborn matplotlib numpy pandas scikit-learn
   ```
4. **Load the dataset** using `seaborn.load_dataset("penguins")` and clean it by removing any missing values.
5. **Run the PCA analysis** by executing the provided code to perform Singular Value Decomposition (SVD) and extract the principal components.
6. **Visualize the results** in a 2D scatter plot showing the first two principal components (PC1 and PC2) with color coding for each penguin species.
7. **Explore the explained variance** with a Scree plot to see how much variance each principal component captures.
