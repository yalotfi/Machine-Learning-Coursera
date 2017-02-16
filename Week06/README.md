# Bias and Variance for Regularized Linear Regression

Compute cross validation errors for unregularized and regularized linear regression.  We can visualize the bias/variance tradeoff, guiding strategy on how to improve the model.  In a underfitting case, high bias can prompt feature mapping by fitting some polynomial curve.  More likely to overfit the data, you can regularize to reduce variance.  Finally, you can find the optimal lambda parameter for regularization by plotting a cross validation curve, assessing the error of both the training and validation sets over a range of different lambda values.

# My Code:

* linearRegCostFunction.m

* learningCurve.m

* polyFeatures.m

* validationCurve.m