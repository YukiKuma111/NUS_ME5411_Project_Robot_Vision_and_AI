close all;
clear;
clc;
%% Task 1
%% 1.1 Display the original image on screen.
global h w c gy_frequency gy_index_grayscales gray_im

im = imread("charact2.bmp");
figure;
imshow(im);
title('Original Input Image: charact2.bmp');

%% 1.2 Experiment with contrast enhancement of the image.
% Convert from RGB to gray
gray_im = rgb2gray(im);

% Apply contrast enhancement to gray image
gy_imadjust = imadjust(gray_im);
gy_histeq = histeq(gray_im);
gy_adapthisteq = adapthisteq(gray_im);

figure;
montage({gray_im,gy_imadjust,gy_histeq,gy_adapthisteq},"Size",[1 4])
title("Original Image and Enhanced Images using " + ...
    "Imadjust, Histeq, and Adapthisteq");

%% 1.3.1 Reproduction
[h,w,c] = size(im);
[gy_frequency, gy_index_grayscales] = imhist(gray_im);

% imadjust
[rpd_imadjust_im, sorted_grayscale, imadjust_grayscale, ...
    lower_grayscale, upper_grayscale] = reproduce_imadjust();
% histeq
[rpd_histeq_im, histeq_mapping] = reproduce_histeq();
% adapthisteq
rpd_adapthisteq_im = reproduce_adapthisteq();

%% 1.3.2 Display and comparison
% imhist
figure;
subplot(1,4,1);
imhist(gray_im);
title("Histogram of Original Gray Image");
ylim([0,40000]);
subplot(1,4,2);
imhist(gy_imadjust);
title("Histogram After Imadjust");
ylim([0,40000]);
subplot(1,4,3);
imhist(gy_histeq);
title("Histogram After Histeq");
ylim([0,40000]);
subplot(1,4,4);
imhist(gy_adapthisteq);
title("Histogram After Adapthisteq");
ylim([0,40000]);

% imadjust mapping
figure;
subplot(3,2,1)
imshow(gray_im);
title("Original Gray Image");
subplot(3,2,3)
imshow(gy_imadjust);
title("Original Imadjust");
subplot(3,2,5);
imshow(rpd_imadjust_im);
title("Reproduction of Imadjust");
subplot(3,2,[2,4,6]);
plot(sorted_grayscale, imadjust_grayscale, '.');
title('Imadjust Mapping');
xlabel('Original Input Pixel Grayscale');
ylabel('Imadjust Output Pixel Grayscale');
hold on;
plot([lower_grayscale, lower_grayscale], [0, 255], ...
    '--r', 'LineWidth', 1.5);
plot([upper_grayscale, upper_grayscale], [0, 255], ...
    '--g', 'LineWidth', 1.5);
legend('Imadjust Mapping', '1% Lower Bound', ...
    '99% Upper Bound', 'Location', 'southeast');
hold off;

% histeq mapping
figure;
subplot(3,2,1)
imshow(gray_im);
title("Original Gray Image");
subplot(3,2,3)
imshow(gy_histeq);
title("Original Histeq");
subplot(3,2,5);
imshow(rpd_histeq_im);
title("Reproduction of Histeq");
subplot(3,2,[2,4,6]);
gray_1d= gray_im(:);
gray_histeq_1d = gy_histeq(:);
hold on;
plot(0:255,round(cumsum(imhist(gray_im))/numel(gray_im)*255),"g.")
scatter(gray_1d, gray_histeq_1d, 10, "blue");
plot(0:255,histeq_mapping,"r.")
title('Histeq Mapping');
xlabel('Original Input Pixel Grayscale');
ylabel('Histeq Output Pixel Grayscale');
legend('Original Gray Image Mapping', 'Original Histeq Mapping', ...
    'Reproduced Histeq Mapping', 'Location', 'southeast');
hold off;

% adapthisteq result compare
figure;
subplot(2,3,1)
imshow(gray_im);
title("Original Gray Image");
subplot(2,3,2);
imshow(gy_adapthisteq);
title("Original Adapthisteq");
subplot(2,3,3);
imshow(rpd_adapthisteq_im);
title("Reproduce Adapthisteq");
subplot(2,3,4);
imhist(gray_im);
ylim([0,40000]);
subplot(2,3,5);
imhist(gy_adapthisteq);
ylim([0,40000]);
subplot(2,3,6);
imhist(rpd_adapthisteq_im);
ylim([0,40000]);

%% Task 2
%% 2.1 Implement and apply a 5x5 averaging filter to the image.
filter_size = 5;
averaging_filter_3 = ones(filter_size) / (filter_size^2);
smoothed_img_5x5 = imfilter(gray_im, averaging_filter_3, 'replicate');

%% 2.2 Experiment with filters of different sizes.
filter_size_2 = 2;
averaging_filter_2 = ones(filter_size_2) / (filter_size_2^2);
smoothed_img_2x2 = imfilter(gray_im, averaging_filter_2, 'replicate');

filter_size_10 = 10;
averaging_filter_10 = ones(filter_size_10) / (filter_size_10^2);
smoothed_img_10x10 = imfilter(gray_im, averaging_filter_10, 'replicate');

filter_size_50 = 50;
averaging_filter_50 = ones(filter_size_50) / (filter_size_50^2);
smoothed_img_50x50 = imfilter(gray_im, averaging_filter_50, 'replicate');

%% 2.3 Compare and comment on the results of the respective image smoothing methods.
figure;
subplot(3,2,[1,2]);
imshow(gray_im);
title('Original Image');
subplot(3,2,3);
imshow(smoothed_img_2x2);
title('Smoothed with 2x2 Filter');
subplot(3,2,4);
imshow(smoothed_img_5x5);
title('Smoothed with 5x5 Filter');
subplot(3,2,5);
imshow(smoothed_img_10x10);
title('Smoothed with 10x10 Filter');
subplot(3,2,6);
imshow(smoothed_img_50x50);
title('Smoothed with 50x50 Filter');

%% 2.4 Custom filter
[diy_filter_result, diy_filter_3d, diy_filter_2d] = custom_imfilter(im, 5);

figure;
subplot(2,2,1);
imshow(im);
title("Original Image");
subplot(2,2,2);
imshow(diy_filter_result);
title("DIY Filtered Image");
subplot(2,2,3);
mesh(diy_filter_3d);
title("DIY Filter in 3D");
subplot(2,2,4);
imshow(diy_filter_2d);
title("Magnified DIY Filter in 2D");

%% Task 3
%% 3.1 Implement and apply a high-pass filter on the image in the frequency domain.
% Perform Fourier Transform
ft = fft2(gray_im);
ft_shift = fftshift(ft);

% Create a meshgrid for frequency domain coordinates
u = floor(-h/2):floor(h/2-1);
v = floor(-w/2):floor(w/2-1);
[U, V] = meshgrid(v, u);

% Compute the distance from the center
distance2center = sqrt(U.^2 + V.^2); 

% Create the high-pass filter mask by energy ratio
ft_magnitude = abs(ft_shift).^2;
energy_cumsum = cumsum(ft_magnitude(:));
total_energy = sum(ft_magnitude(:));
energy_ratio = energy_cumsum / total_energy;
target_ratio = 0.9; % Customizable
idx = find(energy_ratio >= target_ratio, 1);
[row, col] = ind2sub([h,w], idx);
target_distance = distance2center(row, col);
high_pass_mask = double(distance2center > target_distance);

% Apply the high-pass filter
ft_filtered = ft_shift .* high_pass_mask;

% Perform Inverse Fourier Transform
ft_filtered_shifted = ifftshift(ft_filtered);
high_pass_filtered_im = real(ifft2(ft_filtered_shifted));
less_than_zero_mark = high_pass_filtered_im < 0;
high_pass_filtered_im(less_than_zero_mark) = 0;
high_pass_filtered_im = uint8(round(high_pass_filtered_im));

%% 3.2 Compare and comment on the results and the resultant image in the spatial domain.
figure;
subplot(2,1,1);
imshow(gray_im);
title('Original Image');
subplot(2,1,2);
imshow(high_pass_filtered_im);
title('High-Pass Filtered Image');

%% 4. Create a sub-image that includes the middle line – HD44780A00.
middle_line = ceil(h/2:h);
sub_im = im(middle_line, :, :);
sub_gray_im = gray_im(middle_line, :, :);
[sub_h, sub_w, sub_c] = size(sub_im);
figure;
subplot(2,1,1);
imshow(sub_im);
title(sprintf(['Sub-Image That Includes the Middle Line ' ...
    '(size: %d x %d)'], sub_h, sub_w));
subplot(2,1,2);
imshow(sub_gray_im);
title("Grayscale the Input Image");

%% 5. Convert the sub-image into a binary image.
clc;
binary_im = imbinarize(sub_gray_im);
rpd_binary_im = reproduce_imbinarize(sub_gray_im);

figure;
subplot(3,1,1);
imshow(gray_im);
title('Original Gray Image');
subplot(3,1,2);
imshow(binary_im);
title('Binary Image of the Sub-Image');
subplot(3,1,3);
imshow(rpd_binary_im);
title('Custom Binary Image of the Sub-Image');

%% Appendix:
% Reproduce imadjust
function [result_im, sorted_grayscale,sorted_imadjust_grayscale, ...
            lower_grayscale, upper_grayscale] = reproduce_imadjust()
    global gy_frequency gray_im
    result_im = zeros(size(gray_im));
    
    sorted_grayscale = sort(gray_im(:));
    total_pixels = sum(gy_frequency(:));
    % Contrast limits for the input image (default range: 1% ~ 99%)
    lower_bound = ceil(total_pixels * 0.01);
    upper_bound = floor(total_pixels * 0.99);
    lower_grayscale = sorted_grayscale(lower_bound);
    upper_grayscale = sorted_grayscale(upper_bound);
    % Calculate the relationship of input and output values 
    % (default gamma: 1)
    ratio = double((255-0)) / double((upper_grayscale - lower_grayscale));
    % ---------- Inefficienct ----------
    % for i = 1:h
    %     for j = 1:w
    %         grayscale = gray_im(i, j);
    %         if grayscale <= lower_grayscale
    %             rpd_imadjust(i, j) = 0;
    %         elseif grayscale >= high_intensity
    %             rpd_imadjust(i, j) = 255;
    %         else
    %             rpd_imadjust(i, j) = grayscale * gamma;
    %         end
    %     end
    % end
    % ---------- Modified ----------
    result_im(gray_im <= lower_grayscale) = 0;
    result_im(gray_im >= upper_grayscale) = 255;
    mask = (gray_im > lower_grayscale) & (gray_im < upper_grayscale);
    result_im(mask) = ratio * (gray_im(mask) - lower_grayscale);
    result_im = uint8(result_im);
    sorted_imadjust_grayscale = sort(result_im(:));
end

% Reproduce histeq
function [result_im, look_up_table] = reproduce_histeq()
    global gy_frequency gy_index_grayscales gray_im
    result_im = zeros(size(gray_im));
    % Number of discrete gray levels (default n: 64)
    discrete_gray_levels = 64;
    gray_interval = max(gy_index_grayscales) / discrete_gray_levels;
    % ---------- Inefficienct ----------
    % previous_f = 0;
    % practical_cumulative_frequency = [];
    % for i = 1:256
    %     practical_cumulative_frequency(i,1) = gy_frequency(i) + previous_f;
    %     previous_f = practical_cumulative_frequency(i);
    % end
    % ---------- Modified ----------
    cumulative_frequency = cumsum(gy_frequency);
    cumulative_distribution = cumulative_frequency / max(cumulative_frequency);
    look_up_table = round(round( ...
        cumulative_distribution * 255 / gray_interval) * gray_interval);
    result_im = look_up_table(gray_im + 1);
    result_im = uint8(result_im);
end

% Reproduce adapthisteq
function [result_im] = reproduce_adapthisteq()
    global h w gy_frequency gray_im
    result_im = zeros(size(gray_im));
    % Number of rectangular contextual regions (tiles) into which
    % the image is  divided (default: [8, 8])
    num_tiles = [8, 8];
    % A contrast factor that prevents oversaturation of the image
    clip_limit = 0.01;

    tile_size_row = floor (h / num_tiles(1));
    tile_size_col = floor (w / num_tiles(2));
    max_frequency = round(clip_limit * (tile_size_row * tile_size_col));
    
    for i = 1:num_tiles(1)
        for j = 1:num_tiles(2)
            % get tiles
            row_start = (i-1) * tile_size_row + 1;
            row_end = i * tile_size_row;
            col_start = (j-1) * tile_size_col +1;
            col_end = j * tile_size_col;
            if i == num_tiles(1)
                row_end = h;
            end
            if j == num_tiles(2)
                col_end = w;
            end
            tile = gray_im(row_start:row_end, col_start:col_end);
            selected_range = [min(tile(:)), max(tile(:))];
            % calculate tiles' histeq, and distribute the clipped excess
            % frequence proportionally to the valid pixels
            [tile_frequence, ~] = imhist(tile);
            excess_indexes = tile_frequence > max_frequency;
            
            while any(excess_indexes)
                qualified_indexes = (tile_frequence <= max_frequency) ...
                    & (tile_frequence ~= 0);
                clipped_excess_frequencies = ...
                    tile_frequence(excess_indexes) - max_frequency;
                average_clipped_excess_frequencies = ...
                    round(sum(clipped_excess_frequencies) / ...
                    sum(tile_frequence(qualified_indexes)));
                tile_frequence = tile_frequence + qualified_indexes * ...
                    average_clipped_excess_frequencies;
                tile_frequence(excess_indexes) = max_frequency;
                excess_indexes = tile_frequence > max_frequency;
            end
            cumulative_frequency = cumsum(tile_frequence);
            cumulative_distribution = cumulative_frequency / ...
                max(cumulative_frequency);
            lookup_table = round(cumulative_distribution * ...
                double(selected_range(2) - selected_range(1))) + ...
                double(selected_range(1));
            result_im(row_start:row_end, col_start:col_end) = ...
                lookup_table(double(tile) + 1);
        end
    end

    % bilinear interpolation
    % row
    for i = 2:(num_tiles(1) + 1)
        row = (i - 1) * tile_size_row;
        for j = 2:(w-1)
            result_im(row, j) = round((1/4) * ( ...
                result_im(row-1, j-1) + result_im(row+1, j-1) + ...
                result_im(row-1, j+1) + result_im(row+1, j+1)));
        end
    end
    % col
    for i = 2:(num_tiles(2) + 1)
        col = (i - 1) * tile_size_col;
        for j = 2:(h-1)
            result_im(j, col) = round((1/4) * ( ...
                result_im(j-1, col-1) + result_im(j+1, col-1) + ...
                result_im(j-1, col+1) + result_im(j+1, col+1)));
        end
    end
    result_im = uint8(result_im);
end

% Custom a image filter function
function [result_im, filter_disk, mag_filter_disk] = ...
    custom_imfilter(ori_im, sigma_size)
    global h w c
    result_im = zeros(size(ori_im));
    
    % Generate a filter
    [x, y] = meshgrid(-sigma_size:1:sigma_size, -sigma_size:1:sigma_size);
    xy = [x(:) y(:)];
    mu = [0 0];
    sigma = eye(2) * power(sigma_size,2);
    f = mvnpdf(xy, mu, sigma);
    f = reshape(f, length(x), length(y));
    avg_error = (1 - sum(f(:))) / numel(f);
    filter_disk = f + avg_error;

    % Magnifiy the filter disk for display
    magnification = floor(255 / filter_disk(sigma_size+1, sigma_size+1));
    mag_filter_disk = magnification * filter_disk;
    mag_filter_disk = uint8(mag_filter_disk);
    
    % Extend input image boundary
    extended_im = extend_boundary(ori_im, sigma_size);

    % Filter the image
    for i = 1:h
        extended_end_i = i + sigma_size * 2;
        for j = 1:w
            for color = 1:c
                extended_end_j = j + sigma_size * 2;
                block = double(extended_im( ...
                    i:extended_end_i, j:extended_end_j, color));
                filtered_block = block .* filter_disk;
                result_im(i, j, color) = sum(filtered_block(:));
            end
        end
    end
    result_im = uint8(result_im);

end

% Extend the input image boundaries symmetrically
function [result_im] = extend_boundary(ori_im, radius)
    [h, w, c] = size(ori_im);
    result_im = zeros([h+radius*2,w+radius*2,c], 'uint8');
    [eh, ew, ~] = size(result_im);

    % |1|   2   |3|
    % |4|   5   |6|
    % |7|   8   |9|
    for i = 1:eh
        for j = 1:ew
            if (i >= radius+1) && (i <= h+radius) && (j >= radius+1) ...
                    && (j <= w+radius)% #5
                result_im(i,j,:) = ori_im(i-radius, j-radius,:);
            elseif (i < radius+1) && (j >= radius+1) && (j <= w+radius)% #2
                result_im(i,j,:) = ori_im(radius-i+1, j-radius,:);
            elseif (i > h+radius) && (j >= radius+1) && (j <= w+radius)% #8
                result_im(i,j,:) = ori_im(2*h+radius-i+1, j-radius,:);
            elseif (i >= radius+1) && (i <= h+radius) && (j < radius+1)% #4
                result_im(i,j,:) = ori_im(i-radius, radius-j+1,:);
            elseif (i >= radius+1) && (i <= h+radius) && (j > w+radius)% #6
                result_im(i,j,:) = ori_im(i-radius, 2*w+radius-j+1,:);
            elseif (i < radius+1) && (j < radius+1)% #1
                result_im(i,j,:) = ori_im(radius-i+1, radius-j+1,:);
            elseif (i < radius+1) && (j > w+radius)% #3
                result_im(i,j,:) = ori_im(radius-i+1, 2*w+radius-j+1,:);
            elseif (i > h+radius) && (j < radius+1)% #7
                result_im(i,j,:) = ori_im(2*h+radius-i+1, radius-j+1,:);
            elseif (i > h+radius) && (j > w+radius)% #9
                result_im(i,j,:) = ori_im(2*h+radius-i+1, 2*w+radius-j+1,:);
            end
        end
    end
    result_im = uint8(result_im);

end

% Reproduce imbinarize
function [result_im] = reproduce_imbinarize(ori_im)
    result_im = zeros(size(ori_im), 'logical');
    [counts, index] = imhist(ori_im);
    
    % Calculate omega_1: the cumulative proportion of pixels
    omega_1 = cumsum(counts / sum(counts));

    % Calculate the weighted cumulative sum of gray levels (omega_1 * mu_1)
    % and its total (mu_total)
    mu = cumsum(counts .* (index + 1)) ./ cumsum(counts) .* omega_1;
    nan_indices = isnan(mu);
    mu(nan_indices) = 0;
    mu_total = mu(end);

    % Calculate the variance between classes, and find the maximum
    sigma_b_squared = (mu - omega_1 .* mu_total) .^ 2 ./ ...
        (omega_1 .* (1 - omega_1));
    [max_var, threshold] = max(sigma_b_squared);

    result_im = ori_im > threshold;

end
