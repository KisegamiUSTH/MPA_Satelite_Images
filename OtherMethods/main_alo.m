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

% Run ALO enhancement once
disp('Running ALO optimization...');

% Measure the time for this run
iterationStartTime = tic;

% Apply ALO Optimization (GPU-Enabled)
[alo_enhanced_img, bestFitness] = alo_enhance(input_img);

% Store the runtime for this run
runtime = toc(iterationStartTime);

% Display the best fitness for this run
disp(['Best Fitness for this run: ', num2str(bestFitness)]);
disp(['Runtime for this run: ', num2str(runtime), ' seconds']);

% Save the enhanced image as a .jpg file
output_file_name = 'alo_enhanced_image.jpg';
imwrite(alo_enhanced_img, output_file_name);
disp(['Enhanced image saved as: ', output_file_name]);

% Visualize the final enhanced image
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(alo_enhanced_img, []);
title('ALO Enhanced Image');

disp('Enhancement process completed.');

% Step 3: Evaluate the enhancement using the valuation function
disp('Evaluating the enhancement...');
[psnr_value, ssim_value, mse_value, snr_value, original_entropy, enhanced_entropy, mli_value, ag_value, ci_value] = valuation(input_img, alo_enhanced_img);

% Display evaluation metrics
disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);
disp(['MSE: ', num2str(mse_value)]);
disp(['SNR: ', num2str(snr_value)]);
disp(['Original Entropy: ', num2str(original_entropy)]);
disp(['Enhanced Entropy: ', num2str(enhanced_entropy)]);
disp(['MLI (Mean Local Intensity): ', num2str(mli_value)]);
disp(['AG (Average Gradient): ', num2str(ag_value)]);
disp(['CI (Contrast Improvement): ', num2str(ci_value)]);
