function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
%J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%-----------------------------------------------------%
%--- Compute predictions: Forward Propagation (FP) ---%
%-----------------------------------------------------%
% L-1: Input Layer
a1 = [ones(m, 1) X]; % Add ones to the X training data

% L-2: Hidden Layer's first weight parameters and pass sigmoid
z2 = Theta1 * a1';
a2 = [ones(m, 1), sigmoid(z2)']; % Add Bias unit one

% L-3: Output Layer's econd weight params and pass sigmoid into predictions 
z3 = Theta2 * a2';
FP = sigmoid(z3)';

%-----------------------------------%
%--- Vectorized Cost Computation ---%
%-----------------------------------%
% Recode training labels, y, to binary row vectors
Y = eye(num_labels,num_labels); % 10x10 Identity-Matrix
yBin = Y(y, :); % Logical Index to assign correct vectors

% Unregularized Cost with Initial Params
J = (-1/m) * sum(sum(yBin .* log(FP) + (1-yBin) .* log(1-FP),2));

% Add Regularized Expression
J = J + (lambda/(2*m)) * ...
    (sum(sum(Theta1(:,2:end).^2),2) + sum(sum(Theta2(:,2:end).^2,2)));

%-----------------------%
%--- Backpropagation ---%
%-----------------------%
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
for t = 1:m
    %-- STEP 1: Perform Forward Prop on Each Training Example, t --%
    inputLayerOutputs = [1 X(t, :)]; % a_1
    hiddenLayerOutputs = [1; sigmoid(Theta1 * inputLayerOutputs')]; % a_2
    outputLayerOutputs = sigmoid(Theta2 * hiddenLayerOutputs);
    
    %-- STEP 2: Compute Error of Output layer, delta3 --%
    outputError = outputLayerOutputs' - yBin(t, :);
    
    %-- STEP 3: Compute Error of Hidden Layer, delta2 --%
    hiddenError = (outputError * Theta2) ...
        .* sigmoidGradient([1; Theta1 * inputLayerOutputs'])';
    
    %-- STEP 4: Accumulate Gradients --%
    Theta1_grad = Theta1_grad + hiddenError(2:end)' * inputLayerOutputs;
    Theta2_grad = Theta2_grad + (hiddenLayerOutputs * outputError)';
end

%-- STEP 5: Compute Partial Derivatives --%
% Delta1 = (1/m) * Theta1_grad;
% Delta2 = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad + ...
    (lambda/m) * [zeros(size(Theta1,1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + ...
    (lambda/m) * [zeros(size(Theta2,1), 1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
