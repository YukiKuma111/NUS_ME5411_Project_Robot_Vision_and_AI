close all; 
clear; 
clc;
%% 
% The same preprocessing method as task 7
im = imread("charact2.bmp");
[h, w, c] = size(im);

middle_line = ceil(h / 2:h);
sub_im = im(middle_line, :, :);

gray_im = rgb2gray(im);
sub_gray_im = gray_im(middle_line, :);
im_enhanced = imadjust(sub_gray_im);
denoised_im = medfilt2(im_enhanced, [5 5]);
binary_im = imbinarize(denoised_im);
se_open = strel('disk', 6); 
cleaned_im_open = imopen(binary_im, se_open); 
se_close = strel('disk', 3); 
cleaned_im = imclose(cleaned_im_open, se_close);
inverted_im = imcomplement(cleaned_im);

edges = edge(inverted_im, 'Canny');
[labeled_img, num_objects] = bwlabel(edges);
props = regionprops(labeled_img, 'BoundingBox', 'Area');

%% 
% Create a folder to save extracted characters
if ~exist('characters', 'dir')
    mkdir('characters');
end

% Set filtering criteria (minimum area, width, and height thresholds)
min_width = 70;  
max_width = 200;  
split_width_threshold = 150;

% Display character regions with bounding boxes
figure;
imshow(inverted_im);
title('Rectangle Split Characters');
hold on;

n = 0;
% Iterate through each detected region
for k = 1:num_objects
    thisBB = props(k).BoundingBox;
    width = thisBB(3);
    height = thisBB(4);

    % Filter non-character regions based on width
    if width >= min_width && width <= max_width
        % Crop the character from the binary image
        character = imcrop(inverted_im, thisBB);
        
        % Check if the width exceeds the splitting threshold
        if width > split_width_threshold
            % Split the character in half
            mid_col = round(size(character, 2) / 2);
            
            % Left half
            left_character = character(:, 1:mid_col);
            imwrite(uint8(left_character) * 255, fullfile('characters', sprintf('character_%d.png', n)));
            n = n + 1;
            
            % Right half
            right_character = character(:, mid_col + 1:end);
            imwrite(uint8(right_character) * 255, fullfile('characters', sprintf('character_%d.png', n)));
            n = n + 1;

            % Draw bounding boxes for left and right parts
            rectangle('Position', [thisBB(1), thisBB(2), mid_col, height], 'EdgeColor', 'g', 'LineWidth', 1);
            rectangle('Position', [thisBB(1) + mid_col, thisBB(2), width - mid_col, height], 'EdgeColor', 'g', 'LineWidth', 1);
        else
            % Save the entire character if width is within range
            imwrite(uint8(character) * 255, fullfile('characters', sprintf('character_%d.png', n)));

            % Draw the bounding box for the character
            rectangle('Position', thisBB, 'EdgeColor', 'g', 'LineWidth', 1);
            n = n + 1;
        end
    end
end
hold off;
