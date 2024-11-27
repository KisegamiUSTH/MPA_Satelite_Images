clear;
clc;

% GPU Initialization (Commented Out)
gpuDeviceCount;
dg = gpuDevice;
disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('tile_2_4.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Step 2: Apply MPA Optimization
disp('Running MPA optimization on GPU...');
mpa_enhanced_img = mpaenhance(input_img);

% Save the enhanced image as a .jpg file
output_file_name = 'mpa_enhanced_image.jpg';
imwrite(mpa_enhanced_img, output_file_name);
disp(['Enhanced image saved as: ', output_file_name]);

% Visualize the final enhanced image after MPA
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(mpa_enhanced_img, []);
title('MPA Enhanced');

disp('Enhancement process completed.');

% Step 3: Evaluate the enhancement using the valuation function
disp('Evaluating the enhancement...');
[psnr_value, ssim_value, mse_value, snr_value, original_entropy, enhanced_entropy, mli_value, ag_value, ci_value] = valuation(input_img, mpa_enhanced_img);

% Display the results
fprintf('Evaluation Results:\n');
fprintf('MSE: %.4f\n', mse_value);
fprintf('PSNR: %.4f dB\n', psnr_value);
fprintf('SSIM: %.4f\n', ssim_value);
fprintf('SNR: %.4f dB\n', snr_value);
fprintf('Original Entropy: %.4f\n', original_entropy);
fprintf('Enhanced Entropy: %.4f\n', enhanced_entropy);
fprintf('MLI (Mean Local Intensity): %.4f\n', mli_value);
fprintf('AG (Average Gradient): %.4f\n', ag_value);
fprintf('CI (Contrast Improvement): %.4f\n', ci_value);

disp('Evaluation completed.');
