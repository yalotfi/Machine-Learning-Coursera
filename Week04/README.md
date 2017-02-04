# Multi-Class Classifications

Implement one-vs-all logistic regression to recognize hand-written digits between 0 and 9 (where 0 is represented as 10 for MATLAB indexing).  Each training example is a 20x20 gray-scale image which is unrolled into a 1x400 row-wise vector, each column representing a numeric value.

# Basic Nerual Network

Using the same training data, a three layer NN (one input layer, one hidden layer, and one output layer) can be used to classify the handwritten digits.  Performance-wise, logistic regression had an accuracy of 94% predicting new digits while the simple NN predicted 97.5% of unseen data.

## My Code:
* lrCostFunction.m

..* Logistic Regression cost function

* oneVsAll.m

..* Train a one-vs-all K-class classifier (10 in this case)

* predictOneVsAll.m

..* Predict using the trained model

* predict.m

..* Neural Network prediction function