function [X, imageFiles] = characterDataset(folder)
    % Initialize
    X = [];
    
    % Get all images in the filefolder
    imageFiles = dir(fullfile(folder, '*.png'));
    
    for i = 1:length(imageFiles)
        % Read and preprocess the image
        imageFile = fullfile(folder, imageFiles(i).name);
        img = imread(imageFile);
        
        % transform to gray image
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % get the size of original image
        [h, w] = size(img);
        
        % calculate the padding size
        topBottomPadding = round(h / 8);
        leftRightPadding = round(w / 4);
        
        % use white pixels(255) to padding
        img = padarray(img, [topBottomPadding, leftRightPadding], 255, 'both');
        
        % resize the image to 28x28
        img = imresize(img, [28, 28]);
        
        % transform the image to feature metrix
        imgVector = double(img(:))' / 255;
        X = [X; imgVector];
    end
end
