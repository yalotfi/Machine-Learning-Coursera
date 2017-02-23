# SVM for Spam Classification

In this assignment, we built a spam classifier using SVMs. This includes getting an intuition for how SVMs works with 2D datasets.  Doing so means finding the optimal parameters, C and sigma.  For non-linear classification, a gaussian kernel, based on euclidean distance, can fit more complex decision boundaries to a dataset.

The spam classifier requires a lot of preprocessing, or normalizing, of the text through regular expressions. This can range from lower-casing to word stemming, and so on.  After processing the string data, the words' indices can be mapped as binary {1, 0} features which will be used in training the SVM classifier.