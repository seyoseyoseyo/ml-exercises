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

res = X * theta;

res = res - y;

res_cost = res .^ 2;

res_cost = sum(res_cost) / (2 * m);

temp = theta;

temp(1) = 0;

temp_cost = temp .^ 2;

temp_cost = sum(temp_cost) * lambda / (2 * m);

J = res_cost + temp_cost;

res_grad = (res' * X)';

res_grad = res_grad / m;

temp_grad = temp * lambda / m;

grad = res_grad + temp_grad;


% =========================================================================

grad = grad(:);

end
