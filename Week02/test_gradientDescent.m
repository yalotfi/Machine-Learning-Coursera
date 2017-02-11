function [theta, J_history] = test_gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y);
J_history = zeros(num_iters, 1);

% Vectorized solution
% theta = theta - (alpha/m) * (X' * (X * theta-y));

for iter = 1:num_iters
    
    theta = theta - (alpha/m) * (X' * (X * theta - y));

%     temp1 = theta(2) - (alpha/m) * (X' * (prediction - y));    
%     theta = temp0;
%     theta(2) = temp1;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
