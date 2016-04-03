function [ m, PC ] = pcasvd ( A )
%pcasvd Principal Components Analysis using singular value decomposition.
%   Takes A, an (X by K) matrix, where X = [ dimensionality of features ]
%   and K = [ number of examples ] and computes the principal components,
%   or K eigenvectors of A. The result is an (X by K) eigenmatrix.
[~, K] = size(A);

% Mean normalize the input.
[m, A] = meannormalize(A);

% Calculate the covariance matrix.
Acov = 1 / (K - 1) * (A * A');

% Compute the eigenvectors, or principle components.
[~, ~, PC] = svd(Acov);

% Normalize output.
PC = normc(PC);

end
