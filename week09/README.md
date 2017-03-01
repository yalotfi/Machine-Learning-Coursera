# Anomaly Detection with a Gaussian Distribution

By estimating the mean and variance of a normally distributed dataset with a small proportion of anomalous examples, we can train a model on the probability of some feature. By finding the best threshold, epsilon, by which our model captures the anomalous examples, we can classify future cases as either anomalous, `y = 1` or normal, `y = 0`.

This algorithm works best on data that are normally distributed. If it is skewed or otherwise not representaive of a normal distribution, data transformations can improve performance. Primarily based on domain knowledge, new features can also be constructed to capture specific, anomalous events.

# Recommender Systems (in progress)

# My Code:

* estimateGaussian.m

* selectThreshold.m