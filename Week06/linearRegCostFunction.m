function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% diff = X * theta - y;
% theta1 = [0; theta(2:end, :)];
% penalty = lambda * (theta1' * theta1);
% J = (diff' * diff) / (2*m) + penalty / (2*m);
% grad = (X' * diff + lambda * theta1) / m;


% Compute hypothesis function
pred = X * theta;

% Compute Sqaured Error Cost Function
cost_unreg = 1/(2*m) * sum(sum(((pred - y).^2)));
cost_reg = lambda/(2*m)...
    * theta(2:end)' * theta(2:end);

J = cost_unreg + cost_reg;

% Compute Partial Derivatives for Gradient Descent
grad1 = (1/m) * (X' * (pred - y));
reg = (lambda/m) * theta(2:end);
if size(X, 1) <= 1
    grad = grad1;
else
    grad = [grad1(1); ...
            grad1(2:end) + reg];
end
% =========================================================================

grad = grad(:);

end
