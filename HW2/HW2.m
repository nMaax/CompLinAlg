%% TODO
% - [ ] Sparsity storage of matrices
% - [ ] Implementation of power and inverse power methods
% - [ ] 
%% Initialization and Data Loading
clear; clc; close all;

% Load data from 'Circle.mat' where X contains the data points
load('Circle.mat', 'X'); 

% Plot the data
% figure;
% scatter(X(:, 1), X(:, 2));

%% Matrix Size and Sigma Definition
% Get the size of the data matrix X
[n, m] = size(X);

% Allocate space for the similarity matrix S (n x n)
S = zeros(n, n);
sigma = 1; % Define the scale parameter for the Gaussian similarity

%% Construct Similarity Matrix S
% Compute the similarity matrix S using a Gaussian function
for i = 1:n
    for j = i+1:n
        Xi = X(i, :); % Get the i-th data point
        Xj = X(j, :); % Get the j-th data point
        
        % Compute the Gaussian similarity between Xi and Xj
        S(i, j) = exp(-norm(Xi - Xj)^2 / (2 * sigma^2));
    end
end

% Make the similarity matrix symmetric
S = S + S';

% Plot the sparsity pattern of S
% figure;
% spy(S);

%% Construct K-Nearest Neighbors (KNN)
% Allocate space for the KNN matrix (n x k)

k = 10; % Set the number of neighbors to [10, 20, 40]
KNN = zeros(n, k);

for i = 1:n
    % Get the row of the similarity matrix for the i-th point
    neigh_of_i = S(i, :);
    
    % Get the indices of the k largest values (max similarity)
    [~, knn] = maxk(neigh_of_i, k);
    
    % Store the K nearest neighbors for the i-th point
    KNN(i, :) = knn;
    % Uncomment the following line for debugging output
    % fprintf('%d knn are [%d, %d, %d]\n', i, knn);
end

%% Construct Weighted Adjacency Matrix W and Degree Matrix D
% Allocate space for the weighted adjacency matrix W (n x n)
W = zeros(n, n);
B = zeros(n, n);
for i = 1:n
    % Get the K nearest neighbors of the i-th point
    neigh_of_i = KNN(i, :);
    
    % Assign the corresponding similarity values to the adjacency matrix W
    B(i, neigh_of_i) = ones(1, length(neigh_of_i));
    W(i, neigh_of_i) = S(i, neigh_of_i);
    
    B(neigh_of_i, i) = ones(length(neigh_of_i), 1);
    W(neigh_of_i, i) = S(neigh_of_i, i);
end
D = diag(sum(B));

% Plot the sparsity pattern of D (diagonal matrix)
% figure;
% spy(D);

% Plot the sparsity pattern of W
% figure;
% spy(W);

%% Construct Laplacian Matrix L = D - W
L = D - W;

% Plot the sparsity pattern of L
% figure;
% spy(L);

%% Number of connected components

% Create a graph from the adjacency matrix
G = graph(L);

% Compute the connected components
% conncomp returns the component index for each node
components = conncomp(G);

% The number of connected components is the number of unique component indices
num_components = max(components);

% Display the connected components
disp(['Number of connected components: ', num2str(num_components)]);

%% Final Output
fprintf('*** End ***\n');
