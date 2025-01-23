function [X, Y] = generateDataset(folder, labels)
    % Initialize
    X = [];
    Y = [];
    
    for i = 1:length(labels)
        label = labels{i};
        labelFolder = fullfile(folder, label);
        
        % Get all images in the filefolder
        imageFiles = dir(fullfile(labelFolder, '*.png'));
        
        for j = 1:length(imageFiles)
            % Read and preprocess the image
            imageFile = fullfile(labelFolder, imageFiles(j).name);
            img = imread(imageFile);
            % resize the image to 28x28
            img = imresize(img, [28, 28]);
            
            % transform to gray image
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            
            % transform the image to feature metrix
            imgVector = double(img(:))' / 255;
            X = [X; imgVector];
            Y = [Y; i]; % label
        end
    end
end
