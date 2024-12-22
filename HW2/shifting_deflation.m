function [V,D] = shiftingDeflation(A, lambda, x, M)
    
    n = size(x, 1);
    xpeye1 = x(1) + eye(1, n);
    P1 = eye(n) - 2 * (xpeye1*xpeye1') / norm(xpeye1)^2; % norm = 2(1 + eye * x)
    
    % P1 =  P1' = inv(P1) => P1'P1 = P1P1 = I_n
    B1 = P1 * A * P1;
    b1 = B1(2:end, :);
    A2 = B1(2:end, 2:end);

    
    xpeye2 = x(2) + eye(1, n-1);
    P1 = eye(n) - 2 * (xpeye2*xpeye2') / norm(xpeye2)^2; % norm = 2(1 + eye * x)
    
    % P2 =  P2' = inv(P2) => P2'P2 = P2P2 = I_n-1
    B2 = P2 * A2 * P2;
    b2 = B2(2:end, :);
    A3 = B2(2:end, 2:end);

    ...
end