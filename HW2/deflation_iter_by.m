% *** DEFLATION METHOD ITERATION BY ITERATION FOR UNDERSTANDING THE PROCEDURE *** %

clear; close all; clc;

% Power
max_iter = 100;
rel_tol = 1e-6;

% Def
A = [4, -1, 0, 0; 
    -1, 4, -1, 0; 
    0, -1, 4, -1; 
    0, 0, -1, 3];

M = 3;

% Preallocate matrices and data
n = size(A, 1);
D = zeros(M);
V = zeros(n, M);

% Initial guess for eigenvector v_0 (choose randomly or as [1; 1; ...])
v_0 = ones(n, 1); % or use random initial vector
p = 1; % initial shift for inverse power method (can be adjusted)

% *** ITERATION 1 *** %
% FIND x1, eigvec of A with inv power method
[~, x1] = InversePowerMethod(A, v_0, p, max_iter, rel_tol); % how to choose v_0, p?

xpeye1 = x1 + eye(n, 1);
P1 = eye(n) - 2 * (xpeye1*xpeye1') / norm(xpeye1)^2; % norm = 2(1 + eye * x)

% P1 =  P1' = inv(P1) => P1'P1 = P1P1 = I_n
B1 = P1 * A * P1;
b1 = B1(1, 2:end)';
lambda1 = B1(1, 1); 
D(1,1) = lambda1; % save in D
A2 = B1(2:end, 2:end);
V(:, 1) = x1; % save in V


% *** ITERATION 2 *** %
% FIND x2_bar, eigvec of A2 with inv power method
[~, x2_bar] = InversePowerMethod(A2, v_0(1:end-1), p, max_iter, rel_tol); % how to choose v_0, p?

xpeye2_bar = x2_bar + eye(n-1, 1);
P2_bar = eye(n-1) - 2 * (xpeye2_bar*xpeye2_bar') / norm(xpeye2_bar)^2; % norm = 2(1 + eye * x)

P2 = [1, zeros(1, size(P2_bar, 2)); zeros(size(P2_bar, 1), 1), P2_bar];

% P2 =  P2' = inv(P2) => P2'P2 = P2P2 = I_n-1
B2 = P2 * B1 * P2;
b2 = B2(2, 3:end)';
lambda2 = B2(2, 2); 
D(2, 2) = lambda2; % save in D
A3 = B2(3:end, 3:end);

if lambda1 ~= lambda2
    alpha = - (b1' * x2_bar) / (lambda1 - lambda2);
else
    alpha = 0;
end

x2_bar = [alpha; x2_bar];
x2 = P1 * x2_bar; 
V(:, 2) = x2; % save in V

% *** ITERATION 3 *** %
% FIND x3_bar, eigvec of A3 with inv power method
[~, x3_bar] = InversePowerMethod(A3, v_0(1:end-2), p, max_iter, rel_tol); % how to choose v_0, p?

xpeye3_bar = x3_bar + eye(n-2, 1);
P3_bar = eye(n-2) - 2 * (xpeye3_bar*xpeye3_bar') / norm(xpeye3_bar)^2; % norm = 2(1 + eye * x)

P3 = [eye(2), zeros(2, size(P3_bar, 2)); zeros(size(P3_bar, 1), 2), P3_bar];

% P3 =  P3' = inv(P3) => P3'P3 = P3P3 = I_n-2
B3 = P3 * B2 * P3;
b3 = B3(3, 4:end)';
lambda3 = B2(3, 3);
D(3, 3) = lambda3; % save in D
A4 = B3(4:end, 4:end);

if lambda1 ~= lambda3 && lambda2 ~= lambda3 
    beta    =  - (b2' * x3_bar) / (lambda2 - lambda3);
    alpha   =  - (b1' * P2_bar * [beta; x3_bar]) / (lambda1 - lambda3);
else
    beta = 0;
    alpha = 0;
end

x3_bar = [alpha; beta; x3_bar];
x3 = P1 * P2 * x3_bar; % save in V
V(:, 3) = x3;

% *** CONTINUE ... *** %

disp(D);
disp(V);