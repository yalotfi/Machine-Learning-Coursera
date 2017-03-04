# Anomaly Detection with a Gaussian Distribution

By estimating the mean and variance of a normally distributed dataset with a small proportion of anomalous examples, we can train a model on the probability of some feature. By finding the best threshold, epsilon, by which our model captures the anomalous examples, we can classify future cases as either anomalous, `y = 1` or normal, `y = 0`.

This algorithm works best on data that are normally distributed. If it is skewed or otherwise not representaive of a normal distribution, data transformations can improve performance. Primarily based on domain knowledge, new features can also be constructed to capture specific, anomalous events.

# Recommender Systems

Based on user ratings and features of a given product, like a movie, we can use an algorithm called collaborative filtering (also known as low rank matrix factorization). First, it randomely initializes small values for your prediction paramaters and product features. Then you minimize a cost function via some optimization technique like gradient descent or L-BFGS. Finally, predict the rating based on the trained parameters and features.

Based on a the normal distance of two features, similar products can be determined, as well. In terms of implementation, it helps to vectorize the parameter and feature matrices and it is also important to perform mean normalization of user ratings (not necessary for product features).  Doing so allows for you to deal with users who have not provided any ratings or feedback on any products.

# My Code:

* estimateGaussian.m

* selectThreshold.m

* cofiCostFunc.m