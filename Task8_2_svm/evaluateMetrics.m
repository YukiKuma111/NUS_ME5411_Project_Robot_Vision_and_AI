function [accuracyPerClass, precision, recall, f1] = evaluateMetrics(YTrue, YPred, numClasses, labels)
    % Compute the confusion matrix
    confMat = confusionmat(YTrue, YPred);
    
    % Initialize output metrics
    accuracyPerClass = zeros(1, numClasses);
    precision = zeros(1, numClasses);
    recall = zeros(1, numClasses);
    f1 = zeros(1, numClasses);

    for i = 1:numClasses
        TP = confMat(i, i);  % True Positive
        FP = sum(confMat(:, i)) - TP;  % False Positive
        FN = sum(confMat(i, :)) - TP;  % False Negative
        
        % Calculate accuracy, precision, recall, and F1 score for each class
        accuracyPerClass(i) = TP / sum(confMat(i, :), 'omitnan');  % Accuracy for each class
        precision(i) = TP / (TP + FP + eps);  % Precision
        recall(i) = TP / (TP + FN + eps);     % Recall
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);  % F1 Score
    end

    % Plot confusion matrix
    figure;
    cmChart = confusionchart(confMat, labels);
    title('Confusion Matrix');
    cmChart.XLabel = 'Predicted Class';  % Set X-axis label to "Predicted Class"
    cmChart.YLabel = 'True Class';       % Set Y-axis label to "True Class"

    % Plot accuracy bar chart
    figure;
    b1 = bar(accuracyPerClass);
    title('Accuracy per Class');
    xlabel('Class');
    ylabel('Accuracy');
    ylim([0 1]);
    set(gca, 'XTickLabel', labels); % Set x-axis labels
    % Add numerical labels
    for i = 1:numClasses
        text(i, accuracyPerClass(i), num2str(round(accuracyPerClass(i), 2)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    % Plot precision bar chart
    figure;
    b2 = bar(precision);
    title('Precision per Class');
    xlabel('Class');
    ylabel('Precision');
    ylim([0 1]);
    set(gca, 'XTickLabel', labels); % Set x-axis labels
    % Add numerical labels
    for i = 1:numClasses
        text(i, precision(i), num2str(round(precision(i), 2)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    % Plot recall bar chart
    figure;
    b3 = bar(recall);
    title('Recall per Class');
    xlabel('Class');
    ylabel('Recall');
    ylim([0 1]);
    set(gca, 'XTickLabel', labels); % Set x-axis labels
    % Add numerical labels
    for i = 1:numClasses
        text(i, recall(i), num2str(round(recall(i), 2)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    % Plot F1 score bar chart
    figure;
    b4 = bar(f1);
    title('F1 Score per Class');
    xlabel('Class');
    ylabel('F1 Score');
    ylim([0 1]);
    set(gca, 'XTickLabel', labels); % Set x-axis labels
    % Add numerical labels
    for i = 1:numClasses
        text(i, f1(i), num2str(round(f1(i), 2)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
