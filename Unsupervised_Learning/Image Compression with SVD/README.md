# Singular Value Decomposition (SVD) for Image Compression

## a. Brief Description of the Algorithms Implemented

This implementation demonstrates the use of Singular Value Decomposition (SVD) for image compression. SVD is a matrix factorization technique that decomposes a matrix into three smaller matrices: \(U\), \(\Sigma\), and \(V^T\). In this case, the image matrix is decomposed into these components, and by retaining only the largest singular values in \(\Sigma\) and their corresponding vectors in \(U\) and \(V^T\), we can compress the image with minimal loss of quality. The image is then reconstructed using the reduced SVD components, showing how the compression process retains most of the important information while reducing data size.

## b. Summary of the Dataset(s) Used

The dataset used in this implementation consists of grayscale images. A sample image of a hibiscus flower is provided, but users are encouraged to upload their own images. The image is processed to grayscale for simplicity before applying SVD.

## c. Instructions for Reproducing Your Results

1. **Set up your environment**: This code is intended to run on Google Colab. You can open a new Colab notebook and paste the code into a cell.
   
2. **Upload an image**: Run the code, which will prompt you to upload an image from your local machine. You can upload any black and white image or use the provided hibiscus flower image.

3. **Run the SVD compression**: The code will perform Singular Value Decomposition on the grayscale image, then compress the image by retaining a specified number of singular values (k).

4. **View the Results**: The original and compressed images will be displayed side by side. You can adjust the number of retained singular values (k) to see how compression affects the image quality.
