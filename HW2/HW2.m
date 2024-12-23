%% TODO
% - [ ] Tune inverse power method and solve issues related to badly scaled matrices
% - [ ] Other data-sets
% - [ ] Normalized symmetric Laplacian matrix
% - [X] Implementation of inverse power and deflating methods
% - [X] Different clustering techniques in the eigenspace, with tuning
% - [X] Insert a plot for the elbowgraph of 20 eigenvalues magnitudes
% - [ ] LaTeX report
% - [ ] Sparsity storage of matrices

%% Initialization and Data Loading
clear; clc; close all;

% Choose dataset to load
dataset = 'circle'; % Change this variable to switch between datasets

switch dataset
    case 'circle'
        % Load data from 'Circle.mat' where X contains the data points
        load('Circle.mat', 'X');
        
    case 'spiral'
        % Load data from 'Spiral.mat' where X contains the data points
        load('Spiral.mat', 'X');
        
    otherwise
        error('Unknown dataset.');
end

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
    for j = 1:n
        Xi = X(i, :); % Get the i-th data point
        Xj = X(j, :); % Get the j-th data point
        
        % Compute the Gaussian similarity between Xi and Xj
        S(i, j) = exp(-norm(Xi - Xj)^2 / (2 * sigma^2));
    end
end

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
D = diag(sum(W));

% Plot the sparsity pattern of D (diagonal matrix)
% figure;
% spy(D);

% Plot the sparsity pattern of W
% figure;
% spy(W);

%% Construct Laplacian Matrix L = D - W
L = D - W;

% Plot the sparsity pattern of L
figure;
spy(L);

%% Perform the M smallest eigenvalues and the corrispective eigenvectors

M = 3;
eigen_method = 'deflation'; % Change this variable to switch between methods

tic; % Start the timer
switch eigen_method
    case 'deflation'
        [eigenvectors, small_eigenvalues] = DeflationMethod(L, M*10); % Extract more eigenvalues for elbow plotting
        
    case 'builtin'
        [eigenvectors, small_eigenvalues] = eigs(L, M*10, 'smallestabs'); % Extract more eigenvalues for elbow plotting
        
    otherwise
        error('Unknown method for eigenpairs extraction.');
end
eigencalc_elapsed_time = toc; % Stop the timer
fprintf('Eigenvalue calculation elapsed time: %.4f seconds\n', eigencalc_elapsed_time); % Print the elapsed time

diagonal_selected_eigenvalues = small_eigenvalues(1:M,1:M); % Select the M smallest eigenvalues
U = eigenvectors(:,1:M); % Eigenspace spanned by the smallest eigenvalues on which we will perform clustering

%% Perform clustering on the eigenspace
clustering_method = 'dbscan'; % Change this variable to switch between clustering methods

switch clustering_method
    case 'kmeans'
        % Perform clustering using K-means
        [idx, C] = kmeans(U, M); % No tuning needed for K-means
        
    case 'hierarchical'
        % Perform clustering using Agglomerative Hierarchical Clustering with parameter tuning
        linkage_methods = {'ward', 'single', 'complete', 'average'}; % Linkage methods to test
        best_score = -Inf;   % Initialize the best score
        best_idx = [];       % Initialize the best cluster labels
        best_method = '';    % Initialize the best linkage method
    
        % Tuning the hyperparameters of the hierarchical clustering
        for i = 1:length(linkage_methods)
            method = linkage_methods{i};
            % Compute the linkage matrix
            Z = linkage(U, method);
    
            % Perform clustering
            temp_idx = cluster(Z, 'maxclust', M);
    
            % Evaluate clustering quality (silhouette score)
            if length(unique(temp_idx)) > 1 % Avoid single-cluster results
                score = mean(silhouette(U, temp_idx));
                if score > best_score
                    best_score = score;
                    best_idx = temp_idx;
                    best_method = method;
                end
            end
        end
    
        idx = best_idx; % Use the best clustering result
    
        % Print the best parameters
        fprintf('Best Hierarchical Clustering parameters:\n');
        fprintf('  Linkage Method: %s\n', best_method);
        fprintf('  Silhouette Score: %.4f\n', best_score);
        
        case 'dbscan'
            % Perform clustering using DBSCAN with parameter tuning
            eps_range = 0.005:0.005:0.5; % Range of epsilon values to test
            minPts_range = 5:5:15;   % Range of minPts values to test
            best_score = -Inf;       % Initialize the best score
            best_idx = [];           % Initialize the best cluster labels
            best_eps = NaN;          % Initialize the best epsilon
            best_minPts = NaN;       % Initialize the best minPts
            
            % Tuning the hyperparameters epsilon and minPts
            for epsilon = eps_range
                for minPts = minPts_range
                    % Perform DBSCAN clustering
                    temp_idx = dbscan(U, epsilon, minPts);
    
                    % Evaluate clustering quality (silhouette score)
                    if length(unique(temp_idx)) > 1 % Avoid single-cluster results
                        score = mean(silhouette(U, temp_idx));
                        if score > best_score
                            best_score = score;
                            best_idx = temp_idx;
                            best_eps = epsilon;
                            best_minPts = minPts;
                        end
                    end
                end
            end
    
            idx = best_idx; % Use the best clustering result

            % Print the best parameters
            fprintf('Best DBSCAN parameters:\n');
            fprintf('  Epsilon: %.2f\n', best_eps);
            fprintf('  MinPts: %d\n', best_minPts);
            fprintf('  Silhouette Score: %.4f\n', best_score);
        
    otherwise
        error('Unknown clustering method.');
end

%% Plot of the clustering results

% Plot the original data points colored by cluster assignment
figure; % Create a new figure
gscatter(X(:, 1), X(:, 2), idx); % Scatter plot with colors based on cluster index
xlabel('X-axis'); % Label for the x-axis
ylabel('Y-axis'); % Label for the y-axis
title('Spectral Clustering - Cluster Assignments'); % Title of the plot
grid on; % Add grid for better readability

% Check if a third column (ground truth labels) exists in X
if size(X, 2) >= 3
    % Plot the data points with ground truth labels
    figure; % Create another figure
    gscatter(X(:, 1), X(:, 2), X(:, 3)); % Scatter plot with colors based on ground truth labels
    xlabel('X-axis'); % Label for the x-axis
    ylabel('Y-axis'); % Label for the y-axis
    title('Spectral Clustering - Ground Truth Labels'); % Title of the plot
    grid on; % Add grid for better readability
end

%% Plot the elbow graph of the 20 smallest eigenvalues
figure;
plot(diag(small_eigenvalues), 'o-');
hold on;
plot(1:M, diag(diagonal_selected_eigenvalues), 'ro');
xlabel('Eigenvalue Index');
ylabel('Eigenvalue Magnitude');
title('Elbow Graph of 20 Smallest Eigenvalues');
grid on;

%% Final
fprintf('*** End ***\n');
