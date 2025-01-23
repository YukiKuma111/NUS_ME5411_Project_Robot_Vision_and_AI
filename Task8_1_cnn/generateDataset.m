% Function to generate the dataset
function [X, Y] = generateDataset(folder, labels)
    X = [];  % Initialize input dataset (images)
    Y = [];  % Initialize output dataset (labels)
    
    % Iterate through each label folder
    for i = 1:length(labels)
        label = labels{i};  % Current label
        labelFolder = fullfile(folder, label);  % Path to the label folder
        
        % Get all image files in the folder
        imageFiles = dir(fullfile(labelFolder, '*.png'));  % Assuming image format is PNG
        
        % Iterate through each image file
        for j = 1:length(imageFiles)
            imageFile = fullfile(labelFolder, imageFiles(j).name);  % Full path to the image
            img = imread(imageFile);  % Read the image
            
            % Resize the image to 28x28 and convert to grayscale if it's not already
            img = imresize(img, [28, 28]);  % Resize image
            if size(img, 3) == 3
                img = rgb2gray(img);  % Convert RGB to grayscale
            end
            
            % Convert the image to double and normalize it to [0, 1]
            img = double(img) / 255;
            
            % Add the image to the input dataset
            X(:, :, end+1) = img;
            
            % Create one-hot encoding for the label and add to the output dataset
            oneHot = zeros(length(labels), 1);  % Initialize one-hot vector
            oneHot(i) = 1;  % Set the current label's position to 1
            Y(:, end+1) = oneHot;  % Add the one-hot encoded label
        end
    end
    
    % Remove the initial empty image
    X(:, :, 1) = [];  % Remove the first column which was initially empty
end
