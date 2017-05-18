% Project 2  Classification
%
% Name: Wendong Yuan
% Student Num: 8778806
% Date: Nov 2016
%
% File Name: modelTesting.m
%
function pred = modelTesting(model, X)
if (size(X, 2) == 1)
    X = X';
end

% Initialize dataset for prediction 
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

X1 = sum(X.^2, 2);
X2 = sum(model.X.^2, 2)';
K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
K = gaussianKernel(1, 0,0.1) .^ K;
K = bsxfun(@times, model.y', K);
K = bsxfun(@times, model.alphas', K);
p = sum(K, 2);

% Map predictions to 0 and 1
pred(p >= 0) =  1;
pred(p <  0) =  0;

end

% Gaussian Kernel function
function sim = gaussianKernel(x1, x2, sigma)

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

sim = 0;
sigma = 0.1;

n = length(x1);
for i = 1:n
    sim = sim+(x1(i)-x2(i))*(x1(i)-x2(i));
end
sim = (0-sim)/(2*sigma*sigma);
sim = exp(sim);
    
end


