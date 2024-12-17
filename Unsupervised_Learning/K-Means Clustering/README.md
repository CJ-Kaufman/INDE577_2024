# k-Means Clustering Implementation

## a. Brief Description of the Algorithm

The k-Means clustering algorithm is an unsupervised machine learning technique used for partitioning a dataset into **K distinct, spherical clusters** based on proximity. The algorithm works by initializing **K centroids**, assigning each data point to the nearest centroid, and then recalculating the centroids as the mean of the points assigned to each cluster. This process repeats iteratively until the centroids stabilize and the algorithm converges. K-Means is widely used for clustering tasks but may not perform well when clusters are not spherical or have varying densities.

## b. Summary of the Dataset(s) Used

In this implementation, the **make_blobs** function from the `sklearn.datasets` module is used to generate synthetic data for clustering. The data contains **124 samples** with **2 features** and is created to have **5 clusters** (for a K=5 case). The dataset is used for visualization and testing the clustering algorithm.

The data consists of **unlabeled points**. The labels are added for comparison purposes, but the clustering model is trained without knowing these labels.

## c. Instructions for Reproducing Results

1. Clone or download the repository.
2. Install required dependencies:
   ```bash
   pip install matplotlib numpy pandas seaborn scikit-learn
   ```
3. Run the Jupyter notebook or Python script to execute the k-Means clustering algorithm and visualize the results.
4. Modify the value of **K** to experiment with different numbers of clusters and observe the changes in the cluster formations.
