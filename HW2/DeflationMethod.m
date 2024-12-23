function [V, D] = DeflationMethod(A, M, max_iter, rel_tol)
    % DeflationMethod: Compute the M smallest eigenvalues and eigenvectors of a symmetric matrix
    % Input:
    %   A - symmetric matrix, 
    %   M - number of smallest eigenvalues to return
    %   max_iter - maximum number of iterations for inverse power method 
    %   rel_tol - tolerance for convergence for inverse power method
    % 
    % Output: 
    %   V - eigenvector matrix, 
    %   D - eigenvalue matrix
    

    % Preallocate matrices and data
    [n, m] = size(A); % Size of the matrix
    D = zeros(m); % Eigenvalue matrix
    V = zeros(n, m); % Eigenvector matrix

    % Initial guess for eigenvector
    v_0 = ones(n, 1); % Can use random values instead
    p = 0; % Shift for inverse power method (adjust if needed)
    
    
    % Initial values for iterating
    i = 1; % Iteration variable
    B = A; % Start with the original matrix
    P_total = eye(n); % Start with no reflection
    
    % Iteration
    while i <= m
        
        % *** Reflector and eigenpair computing *** %
        
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
        
        % Extract eigenvector correction
        x_correction = zeros(i-1, 1);
        for j = i-1:-1:1
            lambda_prev = D(j, j);
            if lambda ~= lambda_prev
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

    % Sort the results for returning the smallest
    [~, idx] = sort(diag(D), 'ascend');
    D = D(idx, idx);
    V = V(:, idx);

    % % Return the M smallest eigenvalues and eigenvectors
    D = D(:, 1:M);
    V = V(:, 1:M);

end
