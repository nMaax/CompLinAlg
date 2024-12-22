function [V,D] = DeflationMethod(A, M)

    n = size(x, 1);

    % FIND x1, eigenvect of A with inv power method

    xpeye1 = x1 + eye(1, n);
    P1 = eye(n) - 2 * (xpeye1*xpeye1') / norm(xpeye1)^2; % norm = 2(1 + eye * x)
    
    % P1 =  P1' = inv(P1) => P1'P1 = P1P1 = I_n
    B1 = P1 * A * P1;
    % Other components ... b1...
    lambda1 = B1(1, 1);
    A2 = B1(2:end, 2:end);

    % FIND x2_bar, eigenvect of A2 with inv power method
    
    xpeye2_bar = x2_bar + eye(1, n-1);
    P2_bar = eye(n-1) - 2 * (xpeye2_bar*xpeye2_bar') / norm(xpeye2_bar)^2; % norm = 2(1 + eye * x)
    
    P2 = [1, zeros(1, size(P2_bar, 2)); zeros(size(P2_bar, 1), 1), P2_bar];
    
    % P2 =  P2' = inv(P2) => P2'P2 = P2P2 = I_n-1
    B2 = P2 * B1 * P2;
    % Other components ... b2...
    lambda2 = B2(2, 2);
    A3 = B2(3:end, 3:end);
    
    if lambda1 ~= lambda2
        alpha = - (b1' * x2_bar) / (lambda1 - lambda2);
    else
        alpha = 0;
    end
    
    x2_bar = [alpha; x2_bar];
    x2 = P1 * x2_bar;
    
    % FIND x3_bar, eigenvect of A3 with inv power method
    
    xpeye3_bar = x3_bar + eye(1, n-2);
    P3_bar = eye(n-2) - 2 * (xpeye3_bar*xpeye3_bar') / norm(xpeye3_bar)^2; % norm = 2(1 + eye * x)
    
    P3 = [1, zeros(1, size(P3_bar, 2)); zeros(size(P3_bar, 1), 1), P3_bar];
    
    % P3 =  P3' = inv(P3) => P3'P3 = P3P3 = I_n-2
    B3 = P3 * B2 * P3;
    % Other components ...
    lambda3 = B2(3, 3);
    A4 = B3(4:end, 4:end);
    
    if lambda1 ~= lambda3 && lambda2 ~= lambda3 
        beta    =  - (b2' * x3_bar) / (lambda2 - lambda3);
        alpha   = - (b1' * P2_bar * [beta; x3_bar]) / (lambda1 - lambda3);
    else
        beta = 0;
        alpha = 0;
    end
    
    x3_bar = [alpha; beta; x2_bar];
    x3 = P1 * P2 * x3_bar;
    
    % CONTINUE ...
end