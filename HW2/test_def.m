% Test script for DeflationMethod
close all; clear; clc;

% Parameters for DeflationMethod
max_iter = 1000; % Maximum number of iterations
rel_tol = 1e-8; % Tolerance for convergence

% Generate a symmetric matrix
A = [ % Example symmetric matrix
    6, 2, 1;
    2, 6, 1;
    1, 1, 1;
];
M = size(A, 1); % Number of eigenvalues to compute

% Compute eigenvalues and eigenvectors using DeflationMethod
[V, D] = DeflationMethod(A, M, max_iter, rel_tol);

% Compute eigenvalues and eigenvectors using MATLAB's built-in function
[V_builtin, D_builtin] = eigs(A);

% Sort the results for comparison
[D_deflation, idx_deflation] = sort(diag(D), 'descend');
V_deflation = V(:, idx_deflation);

[D_builtin, idx_builtin] = sort(diag(D_builtin), 'descend');
V_builtin = V_builtin(:, idx_builtin);

% Display results
disp('Eigenvalues from DeflationMethod:');
disp(D_deflation);

disp('Eigenvalues from MATLAB eigs():');
disp(D_builtin);

disp('Eigenvectors from DeflationMethod:');
disp(V_deflation);

disp('Eigenvectors from MATLAB eig():');
disp(V_builtin);

% Validate results (allow for numerical tolerance)
eigenvalue_error = norm(D_deflation - D_builtin);
disp(['Eigenvalue error: ', num2str(eigenvalue_error)]);

eigenvector_error = norm(V_deflation - V_builtin, 'fro');
disp(['Eigenvector error (Frobenius norm): ', num2str(eigenvector_error)]);

% Test conclusion
if eigenvalue_error < rel_tol && eigenvector_error < rel_tol
    disp('Test passed: DeflationMethod is accurate.');
else
    disp('Test failed: DeflationMethod results deviate from MATLAB eig().');
end
