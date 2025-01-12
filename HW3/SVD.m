close all; clear; clc;

% Set random seed for reproducibility
rng(337728);
    
% Create a random matrix A
A = rand(6, 4);  % 6x4 matrix
fprintf("\nMatrix A:\n");
disp(A);
 
fprintf("\n---\n");

% *** My SVD *** %
[U, Sigma, V] = compute_svd(A);

fprintf("\n*** My SVD ***\n");
fprintf("\nMatrix U:\n");
disp(U);

% Check if U is orthogonal
U_orthogonal_check = isequal(round(U' * U, 8), eye(size(U, 2)));
fprintf("Is U an orthogonal matrix? %d\n", U_orthogonal_check);

fprintf("\nMatrix Sigma:\n");
disp(Sigma);

fprintf("\nMatrix V.T:\n");
disp(V');

% Check if V is orthogonal
V_orthogonal_check = isequal(round(V' * V, 8), eye(size(V, 2)));
fprintf("Is V an orthogonal matrix? %d\n", V_orthogonal_check);
% Reconstruction check
reconstruction_check = isequal(round(U * Sigma * V', 8), round(A, 8));
fprintf("Is U * Sigma * V.T == A? %d\n", reconstruction_check);

fprintf("\n---\n");

% *** Builtin SVD *** %
[U_builtin, S_builtin, V_builtin] = svd(A, 'econ');

fprintf("\n*** Builtin SVD ***\n");
fprintf("\nMatrix U:\n");
disp(U_builtin);

fprintf("\nMatrix Sigma:\n");
disp(S_builtin);

fprintf("\nMatrix V.T:\n");
disp(V_builtin');

% *** FUNCTIONS *** %

function [U, Sigma, V] = compute_svd(A)
    % Compute SVD of matrix A
    [m, n] = size(A);
    if m < n
        error("Matrix must satisfy m >= n. Given m = %d, n = %d.", m, n);
    end
    
    [B, ~, H] = bidiagonalize(A);
    
    [~, Q_tilde] = eigs_sorted(B' * B, "descend");
    
    Q = H * Q_tilde;
    C = A * Q;
    
    [U, R, P] = qr(C, "vector");
    Sigma = R;
    V = Q(:, P);

    U = U(:, 1:n);
    Sigma = Sigma(1:n, 1:n);
    V = V(:, 1:n);
end

function [B, P, H] = bidiagonalize(A)
    % Bidiagonalization of matrix A
    [m, n] = size(A);
    if m < n
        error("Matrix must satisfy m >= n. Given m = %d, n = %d.", m, n);
    end
    
    B = A;
    P = eye(m);
    H = eye(n);
    
    for k = 1:n
        % Left Householder transformation
        a = B(k:m, k);
        P_tilde = eye(m);
        P_tilde(k:m, k:m) = householder(a);
        B = P_tilde * B;
        P = P_tilde * P;
        
        % Right Householder transformation
        if k <= n - 2
            b = B(k, k+1:n)';
            H_tilde = eye(n);
            H_tilde(k+1:n, k+1:n) = householder(b);
            B = B * H_tilde;
            H = H * H_tilde;
        end
    end
end

function Px = householder(x)
    % Householder transformation
    v = x(:);
    sigma = sign(v(1)) * norm(v);
    u = v + sigma * eye(numel(v), 1);
    u = u / norm(u);
    Px = eye(numel(u)) - 2 * (u * u');
end

function [eigenvalues, eigenvectors] = eigs_sorted(A, order)
    % Compute eigenvalues and eigenvectors
    [eigenvectors, D] = eig(A);
    eigenvalues = diag(D);
    
    % Sort eigenvalues and eigenvectors
    [~, idx] = sort(eigenvalues, order);

    eigenvalues = diag(eigenvalues(idx));
    eigenvectors = eigenvectors(:, idx);
end