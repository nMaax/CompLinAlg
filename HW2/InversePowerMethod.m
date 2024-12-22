function [s_eigval, s_eigvec] = InversePowerMethod(A, v_0, p, max_iter, rel_tol)
    %   INVERSEPOWERMETHOD Compute the eigenvalue of A closest to p using the inverse power method
    %   Inputs:
    %       A        - (n x n) real or complex square matrix.
    %       v_0      - Initial guess for the eigenvector (n x 1 vector).
    %       p        - Eigenvalue target, closest to p.
    %       max_iter - Maximum number of iterations allowed.
    %       rel_tol  - Relative tolerance for convergence (stopping criterion).
    %
    %   Outputs:
    %       s_eigval - Approximation of the eigenvalue closest to p.
    %       s_eigvec - Corresponding normalized eigenvector.

    % Check if it's a square matrix, otherwise there won't be any eigenvalues
    [n, m] = size(A);
    if n ~= m
        error('Not square matrix')
    end
    % Normalize the initial eigenvector guess
    s_eigvec = v_0 / norm(v_0);
    
    % Initialize eigenvalue
    s_eigval = inf;
    
    for i = 1 : max_iter
        
        % Solve linear system to find the smallest eigenvector of step i
        s_eigvec_nonorm = (A - p * eye(n))\ s_eigvec;
        % Normalize the resulting vector
        s_eigvec = s_eigvec_nonorm/ norm(s_eigvec_nonorm);
        % Update the eigenvalue using the Rayleigh quotient
        previous_eigval = s_eigval;
        s_eigval = s_eigvec' * A * s_eigvec;
        
        % Check for convergence
        if abs((s_eigval - previous_eigval) / s_eigval) < rel_tol
            return;
        end
        
        % Update the shift to accelerate convergence
        p = s_eigval;
    end
end