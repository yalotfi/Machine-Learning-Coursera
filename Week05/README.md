# Expanding a 3-Layer NN with Backpropagation

Implementing the backpropagation algorithm on the same dataset of handwritten digits from the previous assignment. The basic workflow for training a neural network with backpropagation is as follows:

1. Random initialization of weights close to zero.

2. Perform feedforward propagation, producing a hypothesis function.

3. Compute the error with a cost function.

4. Perform backpropagation, computing the partial derivatives of the cost.

5. Use gradient checking at least once to gain confidence in the performance of backpropagation.

6. Apply an optimzation algorithm to minimize the cost.

# My Code:

* sigmoidGradient.m

* randomInitializeWeights.m

* nnCostFunction.m