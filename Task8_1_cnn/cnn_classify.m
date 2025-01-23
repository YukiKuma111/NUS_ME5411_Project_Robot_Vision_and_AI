function cnn_classify(net, X)
    % cnn_classify performs classification on input samples using a CNN
    %
    % Inputs:
    % - net: The trained convolutional neural network
    % - X: Input data (3D matrix, where each slice represents one sample)
    %
    % This function performs forward propagation through the network and
    % displays the prediction for each input sample.

    % Define the labels for the classification task
    labels = {'0', '4', '7', '8', 'A', 'D', 'H'};

    % Forward propagation through the network
    [~, ~, depth] = size(X);  % Get the number of samples
    net = cnnff(net, X);      % Perform forward propagation
    
    % Compute the predicted labels for each sample
    [~, h] = max(net.o);      % h contains the indices of predicted labels

    % Display each input sample with its predicted label
    figure;
    for i = 1:depth
        subplot(2, 5, i);
        imshow(X(:, :, i));  % Display the input sample
        title(sprintf('Prediction: %s', labels{h(i)}));  % Show the predicted label
    end
end
