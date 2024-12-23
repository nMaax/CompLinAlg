% Test the InversePowerMethod function
close all; clear; clc;

% Define a test matrix A (e.g., a 4x4 matrix)
A = [4, -1, 0, 0; 
    -1, 4, -1, 0; 
    0, -1, 4, -1; 
    0, 0, -1, 3];

% Define an initial guess for the eigenvector (non-zero vector)
v_0 = [1; 1; 1; 1];

% Define a target eigenvalue close to which we want to find the eigenvalue
p = 2;

% Define maximum number of iterations and relative tolerance for convergence
max_iter = 100;
rel_tol = 1e-6;

% Call the InversePowerMethod function
[s_eigval, s_eigvec] = InversePowerMethod(A, v_0, p, max_iter, rel_tol);

% Display the results
fprintf('Approximate Eigenvalue: %.6f\n', s_eigval);
fprintf('Corresponding Eigenvector: \n');
disp(s_eigvec);

% Verify the result with A * x = lambda * x
Ax = A * s_eigvec;             % Compute A * x
lambda_x = s_eigval * s_eigvec; % Compute lambda * x

% Display the verification result
fprintf('A * x (matrix-vector multiplication): \n');
disp(Ax);
fprintf('lambda * x (eigenvalue times eigenvector): \n');
disp(lambda_x);

% Check if the results are close (using a tolerance)
if norm(Ax - lambda_x) < rel_tol
    disp('Verification successful: A * x ~ lambda * x');
else
    disp('Verification failed: A * x is not close to lambda * x');
end
