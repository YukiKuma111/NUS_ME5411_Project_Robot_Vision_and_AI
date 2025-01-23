close all;
clear;
clc;

% Read the input image
I = imread('charact2.bmp');

% Convert to grayscale
if size(I, 3) == 3
    sub_gray_im = rgb2gray(I);
end

% Display the original image
subplot(3, 2, 1); 
imshow(sub_gray_im);
title('Original Image');

% Enhance the image contrast
im_enhanced = imadjust(sub_gray_im);
subplot(3, 2, 2); 
imshow(im_enhanced);
title('Enhanced Image');

% Apply 5x5 median filtering to remove noise
denoised_im = medfilt2(im_enhanced, [5 5]);
subplot(3, 2, 3); 
imshow(denoised_im);
title('Denoised Image');

% Binarize the image
binary_im = imbinarize(denoised_im);
subplot(3, 2, 4); 
imshow(binary_im);
title('Binarized Image');

% Morphological opening to remove small noise
se_open = strel('disk', 6);
cleaned_im_open = imopen(binary_im, se_open);

% Morphological closing to fill gaps inside characters
se_close = strel('disk', 3); 
BW_eroded = imclose(cleaned_im_open, se_close);
subplot(3, 2, 5); 
imshow(BW_eroded);
title('Cleaned and Filled Image');

% Invert the binary image
inverted_im = imcomplement(BW_eroded);
subplot(3, 2, 6); 
imshow(inverted_im);
title('Inverted Image');

% Apply edge detection
edges = edge(BW_eroded, 'Canny');
figure, imshow(edges), title('Character Outlines');

% Get the height and width of the image
[height, width] = size(BW_eroded);

% Calculate the midline position
midline = floor(height / 2);

% Split the image into the top and bottom halves
BW_top = BW_eroded(1:midline, :);
BW_bottom = BW_eroded(midline+1:end, :);

% Count the number of white pixels in each column (top half)
whitePixelsInColumns_top = sum(BW_top, 1);

% Count the number of white pixels in each column (bottom half)
whitePixelsInColumns_bottom = sum(BW_bottom, 1);

% Find the boundaries in the top half
left_top = find(whitePixelsInColumns_top, 1, 'first');
right_top = find(whitePixelsInColumns_top, 1, 'last');

% Find the boundaries in the bottom half
left_bottom = find(whitePixelsInColumns_bottom, 1, 'first');
right_bottom = find(whitePixelsInColumns_bottom, 1, 'last');

% Calculate the average width of characters in the top half
num_top = 3; % Number of characters in the top half
num_bottom = 10; % Number of characters in the bottom half

width_top = (right_top - left_top) / num_top; % Average width of characters in the top half
width_bottom = (right_bottom - left_bottom) / num_bottom; % Average width of characters in the bottom half

% Label connected components in the binary image
cc = bwconncomp(BW_eroded);
labeled_img = labelmatrix(cc);
rgb_label = label2rgb(labeled_img, 'jet', 'k', 'shuffle');
stats = regionprops(cc, 'BoundingBox');

% Split contours that are too wide
for i = 1:length(stats)
    bbox = stats(i).BoundingBox;
    bbox_width = bbox(3);
    
    % Determine if the contour is in the top or bottom half
    if bbox(2) < midline
        avg_width = width_top;
        region = BW_top;
    else
        avg_width = width_bottom;
        region = BW_bottom;
    end
    
    % If the width is more than 1.5 times the average, split it
    if bbox_width > 1.5 * avg_width
        % Calculate the number of splits
        n = round(bbox_width / avg_width);
        n = max(n, 2); % Ensure at least two splits

        % Calculate the width of each part
        part_width = bbox_width / n;

        % Update the region in the top or bottom half
        if bbox(2) < midline
            % Split the contour into parts
            for j = 1:n-1
                split_pos = floor(bbox(1) + j * part_width-3);
                region(:, split_pos:split_pos+1) = 0;
            end
            BW_top = region;
        else
            % Split the contour into parts
            for j = 1:n-1
                split_pos = floor(bbox(1) + j * part_width);
                region(:, split_pos:split_pos+1) = 0;
            end
            BW_bottom = region;
        end
    end
end

% Merge the top and bottom halves
BW_eroded(1:midline, :) = BW_top;
BW_eroded(midline+1:end, :) = BW_bottom;

% Set a minimum threshold for contour area
min_area_threshold = 20;
% Recompute connected components
cc_new = bwconncomp(BW_eroded);
% Remove contours smaller than the threshold
for i = cc_new.NumObjects:-1:1
    if numel(cc_new.PixelIdxList{i}) < min_area_threshold
        cc_new.PixelIdxList(i) = [];
        cc_new.NumObjects = cc_new.NumObjects - 1;
    end
end

% Generate a labeled matrix for the filtered connected components
labeled_img_new = labelmatrix(cc_new);
rgb_label_new = label2rgb(labeled_img_new, 'jet', 'k', 'shuffle');
figure, imshow(rgb_label_new), title('Segmented and Labeled Characters');
