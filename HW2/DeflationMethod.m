function [V, D] = DeflationMethod(A, M, max_iter, rel_tol)
    % Compute the M smallest eigenvalues and eigenvectors of a symmetric matrix using the deflation method.
    %
    % Inputs:
    %   A        - Symmetric matrix (n x n)
    %   M        - Number of smallest eigenvalues to return (scalar)
    %   max_iter - Maximum number of iterations for the inverse power method (scalar)
    %   rel_tol  - Tolerance for convergence for the inverse power method (scalar)
    %
    % Outputs:
    %   V - Matrix containing the eigenvectors (n x M)
    %   D - Diagonal matrix containing the eigenvalues (M x M)

    % Set default values for optional parameters for Inverse Power Method
    if nargin < 3
        max_iter = 100; % Default maximum number of iterations
    end
    if nargin < 4
        rel_tol = 1e-6; % Default relative tolerance
    end

    % Preallocate matrices and data
    [n, m] = size(A); % Size of the matrix

    % Check if the matrix is square
    if n ~= m
        error('Matrix must be square');
    end

    D = zeros(n); % Eigenvalue matrix
    V = zeros(n, n); % Eigenvector matrix

    % Initial guess for eigenvector
    v_0 = ones(n, 1); % Can use random values instead
    p = 0; % Shift for inverse power method
    
    
    % Initial values for iterating
    i = 1; % Iteration variable
    B = A; % Start with the original matrix
    P_total = eye(n); % Start with no reflection
    
    % Iteration
    while i <= min(n, M)
        
        % *** Reflector and eigenvalue computing *** %
        
        % Pick the submatrix Ai from B for the following iteration
        Ai = B(i:end, i:end);
        
        % Extracting eigenvector from the deflated matrix
        [~, x_bar] = InversePowerMethod(Ai, v_0(1:end-(i-1)), p, max_iter, rel_tol); % how to choose v_0, p?

        % Build Householder reflector
        xbareye = x_bar + eye(n-(i-1), 1);
        P_bar = eye(n-(i-1)) - 2 * (xbareye * xbareye') / sum(xbareye .^ 2);
        P = blkdiag(eye(i-1), P_bar);
        
        % Extract one eigenvalue
        B_prev = B; % Need later eigenvector
        B = P * B * P;
        lambda = B(i, i);
        D(i, i) = lambda; % Save the eigenvalue

        p = lambda; % Update the shift to help convergence
        
        % *** Eigenvector correction *** %
        
        % Compute the correction to the eigenvector
        x_correction = zeros(i-1, 1);
        for j = i-1:-1:1
            lambda_prev = D(j, j); % Previous eigenvalue
            if lambda ~= lambda_prev
                % Compute the correction
                b_corr = B_prev(j, j+1:end);
                x_correction(j) = - (b_corr * [x_correction(j+1:end); x_bar]) / (lambda_prev - lambda);
            else
                x_correction(j) = 0; % Avoid division by zero
            end
        end
        
        % Construct the corrected eigenvector
        x_corrected = [x_correction; x_bar];
        x = P_total * x_corrected; % Transform back to original space
        V(:, i) = x; % Save the eigenvector
                
        % Update the total house holder reflector
        P_total = P * P_total;
        
        % Next iteration
        i = i+1;
    end

    % Sort the results for returning the smallest eigepairs
    D = D(1:M, 1:M);
    V = V(:, 1:M);
end
