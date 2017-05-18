% Project 3  Clustering
%
% Name: Wendong Yuan
% Student Num: 8778806
% Date: Dec 2016
%
% File Name: Kmeans.m
%
function [C, I, iter] = Kmeans(X, K, maxIter, TOL)

% number of vectors in X
[vectors_num, dim] = size(X);

R = randperm(vectors_num);

% construct indicator matrix
I = zeros(vectors_num, 1);

% construct centers matrix
C = zeros(K, dim);

C(1,:) = X(14,:);
for k=2:K
    max=0;
    for sample=1:50
        diff=0;
        Sindex=R(sample);
        for i=1:k-1
            diff = diff + sum((X(Sindex,:)-C(i,:)).^2);
        end
        if diff>max
            C(k,:) = X(Sindex,:);
            max=diff;
        end
    end
end

% iteration count
iter = 0;

while 1
    % find closest point
    for n=1:vectors_num
        minIdx = 1;
        minVal = norm(X(n,:) - C(minIdx,:), 1);
        for j=1:K
            dist = norm(C(j,:) - X(n,:), 1);
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        
        % assign point to the closter center
        I(n) = minIdx;
    end
    
    % compute centers
    for k=1:K
        C(k, :) = sum(X(find(I == k), :));
        C(k, :) = C(k, :) / length(find(I == k));
    end

    RSS_error = 0;
    for idx=1:vectors_num
        RSS_error = RSS_error + norm(X(idx, :) - C(I(idx),:), 2);
    end
    RSS_error = RSS_error / vectors_num;

    iter = iter + 1;

    if 1/RSS_error < TOL
        break;
    end
    
    if iter > maxIter
        iter = iter - 1;
        break;
    end
end

