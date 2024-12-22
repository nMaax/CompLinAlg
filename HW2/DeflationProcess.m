function [deflated_matrix] = DeflationProcess(A, x)
    [m,n] = size(A);
    if n~=m
        error('Not square matrix')
    end
    P = householder(x);
    B = P*A*P;
    deflated_matrix = B(2:n,2:n);
end