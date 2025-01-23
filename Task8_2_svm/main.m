clear all;

% Define dataset paths and SVM parameters
trainFolder = 'dataset/train';
testFolder = 'dataset/test';
labels = {'0', '4', '7', '8', 'A', 'D', 'H'};
C = 1;  % Penalty parameter

% Load and process datasets
[XTrain, YTrain] = generateDataset(trainFolder, labels);
[XTest, YTest] = generateDataset(testFolder, labels);

% Train SVM model
model = trainSVM(XTrain, YTrain, length(labels), C);
% model = trainSVM_quadprog(XTrain, YTrain, length(labels), C);

% Predict and evaluate
YPred = predictSVM(model, XTest);
[accuracy, precision, recall, f1] = evaluateMetrics(YTest, YPred, length(labels), labels);

% Output results
fprintf('Average Accuracy: %.2f\n', mean(accuracy));
fprintf('Average Precision: %.2f\n', mean(precision));
fprintf('Average Recall: %.2f\n', mean(recall));
fprintf('Average F1 Score: %.2f\n', mean(f1));

charactersFolder = 'data';
% Load images from the characters folder as a test set and predict
[XCharactersTest, imageFiles] = characterDataset(charactersFolder);
YPredCharacters = predictSVM(model, XCharactersTest);

% Display images with predicted labels
figure;
for i = 1:length(imageFiles)
    img = reshape(XCharactersTest(i,:), [28, 28]);
    
    subplot(2, 5, i);
    imshow(img, []);
    title(['Pred: ', labels{YPredCharacters(i)}]);
end