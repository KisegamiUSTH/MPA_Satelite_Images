clear;
clc;

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('slice_3_3.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Initialize figure for displaying results
figure;
methods = {'Original', 'MPA', 'ALO', 'DA', 'GA', 'MVO', 'SCA', 'SSA', 'WOA'};
results = {input_img};  % Store the original image as the first result
penalty_values = [NaN];  % Initialize penalty storage with NaN for the original image

% Display the original image first
subplot(3, 3, 1);
imshow(input_img, []);
title('Original Image');

% Run all optimizations and store results
for i = 2:length(methods)
    method_name = methods{i};
    disp(['Running ', method_name, ' optimization...']);
    
    switch method_name
        case 'MPA'
            [results{i}, penalty] = mpaenhance(input_img);
        case 'ALO'
            [results{i}, penalty] = alo_enhance(input_img);
        case 'DA'
            [results{i}, penalty] = da_enhance(input_img);
        case 'GA'
            [results{i}, penalty] = ga_enhance(input_img);
        case 'MVO'
            [results{i}, penalty] = mvo_enhance(input_img);
        case 'SCA'
            [results{i}, penalty] = sca_enhance(input_img);
        case 'SSA'
            [results{i}, penalty] = ssa_enhance(input_img);
        case 'WOA'
            [results{i}, penalty] = woa_enhance(input_img);
    end
    
    % Store the penalty value
    penalty_values(i) = penalty;
    
    % Display the enhanced image
    subplot(3, 3, i);
    imshow(results{i}, []);
    title([method_name, ' Enhanced']);
end

% Evaluate all enhancements and display metrics
metrics = [];
for i = 2:length(methods)  % Skip evaluation for the original image
    [psnr_value, ssim_value, mse_value, snr_value] = valuation(input_img, results{i});
    
    % Display and store the metrics, using the penalty as 'Invalid Pixels'
    fprintf('%s - PSNR: %.4f, SSIM: %.4f, MSE: %.4f, SNR: %.4f, Invalid Pixels (Penalty): %d\n', ...
        methods{i}, psnr_value, ssim_value, mse_value, snr_value, penalty_values(i));
    
    % Append the metrics and penalty to the table
    metrics = [metrics; mse_value, psnr_value, ssim_value, snr_value, penalty_values(i)];
end

% Display the metrics table
disp('Metrics Table:');
disp(metrics);
