# K-Means Clustering

In this exercise, K-means Clustering is used to compress an image represented by a 24-bit scheme (8 bits for red, green, and blue).  K-means can be used to compress the pixel data to 16 colors, where only 4 bits are needed to represent possible values.

K-means is an iterative, two step algorithm. After randomely initializing K number of centroids, each training example is assigned to one based on a squared distance calculation. By computing the mean of the current centroid assignments, you can move centroids to more optimal positions and repeat over each iteration.

# Principle Component Analysis

PCA is a dimension reduction algorithm used to efficiently train learning algorithms by saving disk space and computational work. It is also used to visualize high dimensional data otherwise too difficult to represent in a 2D or 3D space. It is a good idea to normalize and/or scale your features before implementing PCA.

The algorithm is a two step process:

1. Compute the covariance matrix of the data.

2. Compute the eigenvectors of that matrix.

Projecting the data along these vectors effectively map, for example, a 3D set of data along a 2D plane or a 2D set along a 1D line. Of course, PCA is generally used to reduce data with much larger dimensionality. Another important characteristic of PCA is that the projected data can be reformed into its original shape, effectively recovering the data. Data compression is an important application here.

# My Code:

* findClosestCentroids.m

* computeCentroids.m

* pca.m

* projectData.m

* recoverData.m