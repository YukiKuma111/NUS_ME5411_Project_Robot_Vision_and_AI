function [er, bad] = cnntest(net, x, y, labels)
    % Forward pass through the network
    net = cnnff(net, x);
    
    % Find the predicted labels for each sample
    [~, h] = max(net.o); % h is the index of the predicted label
    [~, a] = max(y);     % a is the index of the actual label
    
    % Find the samples that were predicted incorrectly
    bad = find(h ~= a);
    
    % Calculate the error rate
    er = numel(bad) / size(y, 2);
    
    % Initialize variables for performance metrics
    numClasses = length(labels);
    truePositives = zeros(1, numClasses);
    falsePositives = zeros(1, numClasses);
    falseNegatives = zeros(1, numClasses);

    % Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for each class
    for i = 1:numClasses
        truePositives(i) = sum((h == i) & (a == i));  % Correct predictions for class i
        falsePositives(i) = sum((h == i) & (a ~= i)); % Incorrect predictions for class i
        falseNegatives(i) = sum((h ~= i) & (a == i)); % Missed predictions for class i
    end

    % Compute accuracy, precision, recall, and F1 score for each class
    accuracy = (truePositives + (size(y, 2) - (truePositives + falsePositives + falseNegatives))) / size(y, 2);
    precision = truePositives ./ (truePositives + falsePositives + eps);  % Add small epsilon to avoid division by zero
    recall = truePositives ./ (truePositives + falseNegatives + eps);
    f1 = 2 * (precision .* recall) ./ (precision + recall + eps);  % F1 score computation
    
    % Display the average statistics
    fprintf('Average Accuracy: %.2f\n', mean(accuracy) * 100);
    fprintf('Average Precision: %.2f\n', mean(precision) * 100);
    fprintf('Average Recall: %.2f\n', mean(recall) * 100);
    fprintf('Average F1 Score: %.2f\n', mean(f1) * 100);
    
    % Create a figure for the bar plots
    figure;

    % Plot Accuracy bar chart
    subplot(2, 2, 1);
    bar(accuracy);
    title('Accuracy');
    xlabel('Class');
    ylabel('Accuracy');
    xticks(1:numClasses);
    xticklabels(labels);

    % Plot Precision bar chart
    subplot(2, 2, 2);
    bar(precision);
    title('Precision');
    xlabel('Class');
    ylabel('Precision');
    xticks(1:numClasses);
    xticklabels(labels);

    % Plot Recall bar chart
    subplot(2, 2, 3);
    bar(recall);
    title('Recall');
    xlabel('Class');
    ylabel('Recall');
    xticks(1:numClasses);
    xticklabels(labels);

    % Plot F1 Score bar chart
    subplot(2, 2, 4);
    bar(f1);
    title('F1 Score');
    xlabel('Class');
    ylabel('F1 Score');
    xticks(1:numClasses);
    xticklabels(labels);

    % Adjust spacing between subplots
    sgtitle('Performance Metrics for Each Class');

    % Create a confusion matrix figure
    figure;
    confusionMat = confusionmat(a, h);  % Generate confusion matrix
    confusionchart(confusionMat, labels);  % Plot confusion matrix
    title('Confusion Matrix');
end
