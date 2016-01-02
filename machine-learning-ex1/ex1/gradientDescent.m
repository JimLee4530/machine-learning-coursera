function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    costsum_theta1 = 0;
    costsum_theta2 = 0;
    for i = 1:m
        costsum_theta1 = costsum_theta1 + (theta(1,1) * X(i,1) + theta(2,1) * X(i,2) - y(i)) * X(i,1);
        costsum_theta2 = costsum_theta2 + (theta(1,1) * X(i,1) + theta(2,1) * X(i,2) - y(i)) * X(i,2);
    end

    theta_1 = theta(1,1) - alpha * costsum_theta1 * 1/m;
    theta_2 = theta(2,1) - alpha * costsum_theta2 * 1/m;
    theta(1,1) = theta_1;
    theta(2,1) = theta_2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
