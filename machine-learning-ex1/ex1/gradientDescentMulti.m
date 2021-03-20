function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % setup theta_t holding matrix
    theta_t = zeros(length(theta), 1);
    fprintf('%d length holding matrix created...\n', length(theta));

    % calculate thetas
    for t = 1:length(theta)
      fprintf('Calculating theta %d for iteration %d ...\n', t, iter);
      theta_t(t, 1) = theta(t) - ((alpha/m) .* sum((X*theta-y).*X(:, t)));
    end
    fprintf('theta calculated as...\n');
    disp(theta_t);

    theta = theta_t;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
