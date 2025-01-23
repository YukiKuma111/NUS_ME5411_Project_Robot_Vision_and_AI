function [X] = loadImageDataset(folder)
    % Get a list of all PNG files in the folder
    imageFiles = dir(fullfile(folder, '*.png'));  % Get the list of PNG files
    numImages = numel(imageFiles);  % Total number of images
    
    % Initialize the dataset array with dimensions 28x28xn
    X = zeros(28, 28, numImages);
    
    % Iterate over each image file and process it
    for i = 1:numImages
        % Read the image file
        img = imread(fullfile(folder, imageFiles(i).name));
        
        % Convert to grayscale if the image is RGB
        if size(img, 3) == 3
            img = rgb2gray(img);  % Convert RGB to grayscale
        end
        
        % Get the original dimensions of the image
        [h, w] = size(img);
        
        % Calculate padding size
        topBottomPadding = round(h / 8);  % Top and bottom padding as 1/8 of height
        leftRightPadding = round(w / 2);  % Left and right padding as 1/2 of width
        
        % Pad the image using value 1 for padding
        img = padarray(img, [topBottomPadding, leftRightPadding], 1, 'both');
        
        % Optionally reverse and normalize the image
        % X(:, :, i) = 1 - img; % Invert and normalize to [0, 1]
        
        % Resize the image to 28x28 pixels
        img = imresize(img, [28, 28]);
        
        % Store the processed image in the dataset
        X(:, :, i) = img;
    end
end
