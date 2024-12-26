%% TODO
% - [ ] Tune inverse power method and solve issues related to badly scaled matrices
% - [ ] LaTeX report

%% Initialization and Data Loading
clear; clc; close all;

% Choose dataset to load
dataset = 'spiral'; % Change this variable to switch between datasets

switch dataset

    case 'spiral'
        % Load data from 'Spiral.mat' where X contains the data points
        load('Spiral.mat', 'X');
        X = X(:, 1:2); % Keep only the first two columns (ignore the ground truth)

    case 'circle'
        % Load data from 'Circle.mat' where X contains the data points
        load('Circle.mat', 'X');
    
    case 'torus'
        % Load data from 'torus_coordinates.csv' where the columns are x, y, z
        X = readmatrix('torus_coordinates.csv');
          
    otherwise
        error('Unknown dataset.');
end

[n, m] = size(X); % Get the dimensionality of data for later use

M = 3; % Number of clusters to find (ignored if dbscan is used)

assert(M < n, 'The number of clusters must be less than the number of data points.');

%% Construct Similarity Matrix S
% Compute the similarity matrix S using the Gaussian function. 
% Could also be implemented with more naive approaches, but this is the most efficient

% Compute the pairwise squared Euclidean distance matrix, where the (i, j)-th entry is the squared Euclidean distance between the i-th and j-th points. 
% Directly comes from the definition of the Eucledian distance: 
% d(x, y) = sum(xi - yi)^2 = sum(x_i^2 + y_i^2 - 2 * x_i * y_i)
X_square = sum(X.^2, 2);
distance_matrix = X_square + X_square' - 2 * (X * X');

sigma = 1; % Define the scale parameter for the Gaussian similarity
% Compute the similarity matrix S using the Gaussian function
S = exp(-distance_matrix / (2 * sigma^2));

%% Construct K-Nearest Neighbors (kNN) Matrix
k = 10; % Set the number of neighbors to [10, 20, 40]
assert(k < n, 'The number of neighbors must be less than the number of data points.');

[~, kNN] = maxk(S, k, 2); % Find the k nearest neighbors for each point

%% Construct Weighted Adjacency Matrix W and Degree Matrix D
% Allocate index and value arrays for sparse matrices
i_indices = zeros(1, n * k * 2);
j_indices = zeros(1, n * k * 2);
values_W = zeros(1, n * k * 2);

% Fill the index and value arrays for the sparse matrix construction
for i = 1:n

    % Get the k nearest neighbors of the i-th point
    neigh_of_i = kNN(i, :);
    
    % Compute the start and end indices for the i-th row: we will move on the arrays in blocks of 2*k
    % e.g.  __indices[0:k, k+1:2k, 2k+1:3k, ..., (n-1)*k+1:n*k]
    start_idx   = (i - 1)   * k * 2 + 1;
    end_idx     = i         * k * 2;
    
    % Fill the index and value arrays for the sparse matrix construction
    % e.g   i_indices[2k+1:3k] =   [3, 3, 3, ..., 3,    neigh_of_3 ... ];
    %       j_indices[2k+1:3k] =   [neigh_of_3 ... ,    3, 3, 3, ..., 3];
    i_indices(start_idx:end_idx) = [repmat(i, 1, k),    neigh_of_i]; % Row indices for W
    j_indices(start_idx:end_idx) = [neigh_of_i,         repmat(i, 1, k)]; % Column indices for W
    
    % Values for W, generalized for non-symmetric matrices. In symmetric case writing S(neigh_of_i, i) or S(i, neigh_of_i)' is equivalent
    values_W(start_idx:end_idx) =  [S(i, neigh_of_i),    S(neigh_of_i, i)'];
end

W = sparse(i_indices, j_indices, values_W, n, n); % Construct the weighted adjacency matrix W
D = spdiags(sum(W, 2), 0, n, n); % Compute the degree matrix D

%% Construct normalized Laplacian Matrix 
D_inv_sqrt = diag(1 ./ sqrt(diag(D)));

% Compute the unnormalized Laplacian
% L = D - W;

% Compute L_norm = I - D^(-1/2) * W * D^(-1/2)
L = eye(n) - D_inv_sqrt * W * D_inv_sqrt;
L = sparse(L);

% Plot the sparsity pattern of L
figure;
spy(L);

%% Perform the M smallest eigenvalues and the corrispective eigenvectors

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

%% Plot the elbow graph of the 20 smallest eigenvalues
figure;
plot(diag(small_eigenvalues), 'o-');
hold on;
plot(1:M, diag(diagonal_selected_eigenvalues), 'ro');
xlabel('Eigenvalue Index');
ylabel('Eigenvalue Magnitude');
title('Elbow Graph of 20 Smallest Eigenvalues');
legend('All Eigenvalues', 'Selected Eigenvalues');
grid on;

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
        % Perform clustering using DBSCAN with parameter tuning, M is ignored here
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
        error('Unknown clustering method for data in the eigenspace.');
end

%% Plot of the clustering results

% Plot of the clustering results
figure; % Create a new figure
colors = lines(max(idx)); % Use the 'lines' colormap for better visibility on white background

if size(X, 2) == 3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 10, idx, 'filled'); % 3D scatter plot
    colormap(colors); % Apply the colormap
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    axis equal;
    title('Spectral Clustering - Cluster Assignments (3D)');
elseif size(X, 2) == 2
    gscatter(X(:, 1), X(:, 2), idx, colors); % 2D scatter plot, with colormap applied
    xlabel('X-axis');
    ylabel('Y-axis');
    axis equal;
    title('Spectral Clustering - Cluster Assignments (2D)');
end

%% Final
fprintf('*** End ***\n');
