% Define the paths for the training and testing datasets
trainFolder = 'dataset/train';  % Path to the training dataset
testFolder = 'dataset/test';    % Path to the testing dataset

% Define the labels for the dataset
labels = {'0', '4', '7', '8', 'A', 'D', 'H'};

% Generate the training dataset (features and labels)
[XTrain, YTrain] = generateDataset(trainFolder, labels);

% Generate the testing dataset (features and labels)
[XTest, YTest] = generateDataset(testFolder, labels);

% Define the target size for the training dataset
targetSize = 5350;

% Calculate the number of samples to add to reach the target size
numToAdd = targetSize - size(XTrain, 3);

% Randomly select indices to duplicate for augmentation
randomIndices = randi(size(XTrain, 3), [1, numToAdd]);

% Extend the training set by concatenating the randomly chosen images
XTrainAugmented = cat(3, XTrain, XTrain(:, :, randomIndices));
YTrainAugmented = [YTrain, YTrain(:, randomIndices)];  % Ensure labels match the augmented dataset

% Update the training set with the augmented data
XTrain = XTrainAugmented;
YTrain = YTrainAugmented;

% Display the dimensions of the training and testing datasets
disp('XTrain dimensions:');
disp(size(XTrain));
disp('YTrain dimensions:');
disp(size(YTrain));

disp('XTest dimensions:');
disp(size(XTest));
disp('YTest dimensions:');
disp(size(YTest));

% Set the random seed for reproducibility
rand('state', 0);

% Initialize CNN network structure
cnn.layers = {
    struct('type', 'i')                                   % Input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) % Convolutional layer, output 6 feature maps, kernel size 5x5
    struct('type', 's', 'scale', 2)                       % Subsampling (pooling) layer with 2x scaling factor
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)% Convolutional layer, output 12 feature maps, kernel size 5x5
    struct('type', 's', 'scale', 2)                       % Subsampling (pooling) layer with 2x scaling factor
};

% Display the CNN structure
disp_cnn(cnn);

% Set the training options
opts.alpha = 0.5;          % Learning rate
opts.batchsize = 50;     % Number of samples per batch
opts.numepochs = 50;     % Number of epochs (iterations)

% Initialize the CNN with the training data
cnn = cnnsetup(cnn, XTrain, YTrain);

% Train the CNN using the defined options
cnn = cnntrain(cnn, XTrain, YTrain, opts);

% Test the trained model
[er, bad] = cnntest(cnn, XTest, YTest, labels);  % Calculate error rate on the test set

% Load additional character dataset for classification
folder = 'data';  % Path to the character dataset
X_c = loadImageDataset(folder);  % Load the dataset
cnn_classify(cnn, X_c);  % Classify the loaded dataset using the trained CNN

% Save the trained CNN model
save('trained_cnn.mat', 'cnn');

% Output the error rate as a percentage
fprintf('Error rate: %.2f%%\n', er * 100);

% Plot the mean squared error during training
figure;
plot(cnn.rL);
title('Mean Squared Error during Training');
xlabel('Epoch');
ylabel('Mean Squared Error');