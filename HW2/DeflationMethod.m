function [V, D] = DeflationMethod(A, M, max_iter, rel_tol)

    % Preallocate matrices and data
    n = size(A, 1); % Size of the problem
    D = zeros(M); % Eigenvalue matrix
    V = zeros(n, M); % Eigenvector matrix

    % Initial guess for eigenvector
    v_0 = ones(n, 1); % Can use random values instead
    p = 1; % Shift for inverse power method (adjust if needed)
    
    
    % Initial values for iterating
    i = 1; % Iteration variable
    B = A; % Start with the original matrix
    P_total = eye(n); % Start with no reflection
    
    % Iteration
    P_bars = cell(1, M);
    b_vecs = cell(1, M);
    while i <= M
        
        % *** Reflector and eigenpair computing *** %
        
        % Update Ai for the following iteration
        Ai = B(i:end, i:end);
        
        % Extracting eigenvector from the deflated matrix
        [~, x_bar] = InversePowerMethod(Ai, v_0(1:end-(i-1)), p, max_iter, rel_tol); % how to choose v_0, p?

        % Build Householder reflector
        xbareye = x_bar + eye(n-(i-1), 1);
        P_bar = eye(n-(i-1)) - 2 * (xbareye * xbareye') / sum(xbareye .^ 2);
        P_bars{i} = P_bar;
        P = blkdiag(eye(i-1), P_bar);
        
        % Extract one eigenvalue
        B = P * B * P;
        b = B(i, i:end)';
        b_vecs{i} = b;
        lambda = B(i, i);
        D(i, i) = lambda; % Save the eigenvalue
        
        % Extract one eigenvector
        % *** Correct eigenvector ***
%         x_correction = zeros(i-1, 1);
%         for j = i-1:-1:1
%             lambda_prev = D(j, j);
%             if lambda ~= lambda_prev
%                 b_corr = b_vecs{j}' * P_bars{j};
%                 x_correction(j) = - (b_corr * [x_correction(j:end); x_bar]) / (lambda_prev - lambda);
%             else
%                 x_correction(j) = 0; % Avoid division by zero
%             end
%         end
%         
%         % Construct the corrected eigenvector
%         x_corrected = [x_correction; x_bar];
%         x = P_total * x_corrected; % Transform back to original space
%         V(:, i) = x; % Save the eigenvector
        
        % *** Update for next iteration *** %
        
        % Update the total house holder reflector
        P_total = P * P_total;
        
        % Next iteration
        i = i+1;
    end
end
