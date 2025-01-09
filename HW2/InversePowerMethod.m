function [s_eigval, s_eigvec] = InversePowerMethod(A, v_0, p, max_iter, rel_tol)
    % Compute the eigenvalue of A closest to p using the inverse power method
    %   Inputs:
    %       A        - (n x n) square matrix.
    %       v_0      - Initial guess for the eigenvector (n x 1 vector).
    %       p        - Target eigenvalue, the method finds the eigenvalue closest to p.
    %       max_iter - Maximum number of iterations allowed.
    %       rel_tol  - Relative tolerance for convergence (stopping criterion).
    %
    %   Outputs:
    %       s_eigval - Approximation of the eigenvalue closest to p.
    %       s_eigvec - Corresponding normalized eigenvector.

    % Check if it's a square matrix, otherwise there won't be any eigenvalues
    [n, m] = size(A);
    if n ~= m
        error('Not a square matrix')
    end
    
    if isscalar(A)
        s_eigval = A;
        s_eigvec = v_0 / norm(v_0);
        return
    end
    
    % Normalize the initial eigenvector guess
    s_eigvec = v_0 / norm(v_0);
    
    % Initialize eigenvalue
    s_eigval = inf;
    
    for i = 1 : max_iter
        % Solve linear system to find the smallest eigenvector of step i
        try
            s_eigvec_nonorm = (A - p * eye(n)) \ s_eigvec;
        catch
            % If the solve fails, it means p is an eigenvalue
            s_eigval = p;
            s_eigvec = s_eigvec / norm(s_eigvec); % Siamo sicuri?
            return;
        end
        % Normalize the resulting vector
        s_eigvec = s_eigvec_nonorm / norm(s_eigvec_nonorm);
        
        % Update the eigenvalue using the Rayleigh quotient
        previous_eigval = s_eigval;
        s_eigval = s_eigvec' * A * s_eigvec;
        
        % Check for convergence
        if abs((s_eigval - previous_eigval) / s_eigval) < rel_tol
            return;
        end
        
    end
end
