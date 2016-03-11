function A = generatesparseSPDmatrix(n,density)
% Generate a sparse n x n symmetric, positive definite matrix with
%   approximately density*n*n non zeros

A = sprandsym(n,density); % generate a random n x n matrix

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
A = A + n*speye(n);
F = full(A);
save data F;

end