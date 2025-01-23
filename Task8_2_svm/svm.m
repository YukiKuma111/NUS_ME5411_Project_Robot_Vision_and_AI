% Define paths for training and testing folders
trainFolder = 'dataset/train';
testFolder = 'dataset/test';

% Define labels
labels = {'0', '4', '7', '8', 'A', 'D', 'H'};

% Load training data
[XTrain, YTrain] = loadImageData(trainFolder, labels);
XTrain = double(XTrain); % Convert to double precision

% Load testing data
[XTest, YTest] = loadImageData(testFolder, labels);
XTest = double(XTest); % Convert to double precision

% Convert labels to categorical
YTrain = categorical(YTrain);
YTest = categorical(YTest);

% Train multi-class SVM classifier
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', 'BoxConstraint', 1);
SVMModel = fitcecoc(XTrain, YTrain, 'Learners', t);


% Predict and evaluate
YPred = predict(SVMModel, XTest);
[accuracy, precision, recall, f1] = evaluateMetrics(YTest, YPred, length(labels), labels);

% Display results
fprintf('Average Accuracy: %.2f\n', mean(accuracy));
fprintf('Average Precision: %.2f\n', mean(precision));
fprintf('Average Recall: %.2f\n', mean(recall));
fprintf('Average F1 Score: %.2f\n', mean(f1));

% Load and predict characters in 'characters/data' folder
charactersFolder = 'characters/data';
[XCharactersTest, imageFiles] = loadCharacterData(charactersFolder);
XCharactersTest = double(XCharactersTest); % Convert to double precision
YPredCharacters = predict(SVMModel, XCharactersTest);

% Display predictions
figure;
for i = 1:length(imageFiles)
    img = reshape(XCharactersTest(i,:), [28, 28]);
    subplot(2, 5, i);
    imshow(img, []);
    title(['Pred: ', labels{YPredCharacters(i)}]);
end

% Helper function to load images
function [X, Y] = loadImageData(folderPath, labels)
    X = [];
    Y = [];
    for i = 1:length(labels)
        labelFolder = fullfile(folderPath, labels{i});
        imageFiles = dir(fullfile(labelFolder, '*.png')); % Adjust extension as needed
        for j = 1:length(imageFiles)
            img = imread(fullfile(labelFolder, imageFiles(j).name));
            img = imresize(img, [28 28]); % Resize to 28x28 if needed
            if size(img, 3) == 3
                img = rgb2gray(img); % Convert to grayscale if image is RGB
            end
            X = [X; img(:)'];
            Y = [Y; i]; % Store label index as Y
        end
    end
end

% Helper function to load character images
function [X, imageFiles] = loadCharacterData(folderPath)
    imageFiles = dir(fullfile(folderPath, '*.png')); % Adjust extension as needed
    X = [];
    for i = 1:length(imageFiles)
        img = imread(fullfile(folderPath, imageFiles(i).name));
        if size(img, 3) == 3
            img = rgb2gray(img); % Convert to grayscale if image is RGB
        end
        [h, w] = size(img);
        topBottomPadding = round(h / 8);
        leftRightPadding = round(w / 4);
        % use white pixels(255) to padding
        %img = padarray(img, [topBottomPadding, leftRightPadding], 255, 'both');
        img = imresize(img, [28 28]); % Resize to 28x28
        X = [X; img(:)'];
    end
end
