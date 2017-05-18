% Project 2  Classification
%
% Name: Wendong Yuan
% Student Num: 8778806
% Date: Nov 2016
%
% File Name: modelTraining.m
%
function [model] = modelTraining(X, Y, C, kernelFunction, tol, max_passes)

% Parameter for tolerance
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

% Parameter for iterations
if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

m = size(X, 1); % number of rows
n = size(X, 2); % number of columns

% Map 0 to -1
Y(Y==0) = -1;

% Basic parameters
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

X2 = sum(X.^2, 2);
K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
K = gaussianKernel(1, 0, 0.1) .^ K;

while passes < max_passes,
            
    num_changed_alphas = 0;
    for i = 1:m,
        
        % Calculate Ei = f(x(i)) - y(i) 
        E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
        
        if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0)),
            
            % Select random i and j
            j = ceil(m * rand());
            while j == i, 
                j = ceil(m * rand());
            end

            % Calculate Ej = f(x(j)) - y(j).
            E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % Compute L and H
            if (Y(i) == Y(j)),
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H),
                continue;
            end

            % Compute eta
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0),
                continue;
            end
            
            % Compute and clip new value for alpha j
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            
            % Clip
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            % Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol), 
                alphas(j) = alpha_j_old;
                continue;
            end
            
            % Determine value for alpha i
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            % Compute b1 and b2 
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

            if (0 < alphas(i) && alphas(i) < C),
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C),
                b = b2;
            else
                b = (b1+b2)/2;
            end

            num_changed_alphas = num_changed_alphas + 1;

        end
        
    end
    
    if (num_changed_alphas == 0),
        passes = passes + 1;
    else
        passes = 0;
    end

    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
%model.kernelFunction = gaussianKernel;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';

end

% Gaussian Kernel function
function sim = gaussianKernel(x1, x2, sigma)

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
