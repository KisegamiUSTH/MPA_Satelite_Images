clear;
clc;

% GPU Initialization
%gpuDeviceCount;
%dg = gpuDevice;
%disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('slice_3_3.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Run the MVO enhancement once
disp('Running MVO optimization on GPU...');

% Measure the start time
tic;

% Apply MVO Enhancement and retrieve the best fitness
[mvo_enhanced_img, bestFitness] = mvo_enhance(input_img);

% Measure the runtime for this run
runTime = toc;

% Save the enhanced image
output_file_name = 'mvo_enhanced_image.jpg';
imwrite(mvo_enhanced_img, output_file_name);
disp(['Enhanced image saved as: ', output_file_name]);
disp(['Best Fitness: ', num2str(bestFitness) ' | Runtime: ' num2str(runTime) ' seconds']);

% Visualize the final enhanced image after MVO
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(mvo_enhanced_img, []);
title('MVO Enhanced Image');

% Step 3: Evaluate the enhancement using the valuation function
disp('Evaluating the enhancement...');
[psnr_value, ssim_value, mse_value, snr_value] = valuation(input_img, mvo_enhanced_img);

% Display evaluation metrics
disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);
disp(['MSE: ', num2str(mse_value)]);
disp(['SNR: ', num2str(snr_value)]);

disp('MVO enhancement process completed.');
