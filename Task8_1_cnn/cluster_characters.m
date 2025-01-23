function idx = cluster_characters(occupancy_ratios, avg_distances, num_clusters)
    % cluster_characters performs K-means clustering and visualizes the results
    %
    % Inputs:
    % - occupancy_ratios: Array of occupancy ratios for each character
    % - avg_distances: Array of average Euclidean distances for each character
    % - num_clusters: Number of clusters for K-means
    %
    % Output:
    % - idx: Cluster labels for each character

    % Combine occupancy ratios and average distances into 2D data
    data = [occupancy_ratios', avg_distances'];
    
    % Perform K-means clustering
    [idx, C] = kmeans(data, num_clusters);
    
    % Visualize clustering results
    figure;
    scatter(data(:, 1), data(:, 2), 100, idx, 'filled');  % Color-coded scatter plot
    hold on;
    plot(C(:, 1), C(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);  % Plot cluster centers
    xlabel('Occupancy Ratio');
    ylabel('Average Euclidean Distance');
    title(sprintf('K-means Clustering with %d Clusters', num_clusters));
    grid on;
    hold off;
    
    % Display cluster centers
    disp('Cluster centers:');
    disp(C);
    
    % Display cluster assignments
    disp('Cluster assignments:');
    for i = 1:length(idx)
        fprintf('Data point %d (Occupancy Ratio: %.4f, Avg Distance: %.4f) is in cluster %d\n', ...
                i, occupancy_ratios(i), avg_distances(i), idx(i));
    end
    
    % Return cluster labels
end