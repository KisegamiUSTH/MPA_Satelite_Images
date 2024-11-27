clear;
clc;

% GPU Initialization
gpuDeviceCount;
dg = gpuDevice;
disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('slice_3_3.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Start timing the run
tic;

% Run the GWO enhancement
disp('Running GWO optimization on GPU...');
[gwo_enhanced_img, bestFitness] = gwo_enhance(input_img);

% Record runtime for this run
runtime = toc;

% Save the enhanced image as a .jpg file
output_file_name = 'gwo_enhanced_image.jpg';
imwrite(gwo_enhanced_img, output_file_name);
disp(['Enhanced image saved as: ', output_file_name]);

% Display the best fitness for this run
disp(['Best Fitness: ', num2str(bestFitness)]);

% Display the runtime for this run
disp(['Runtime: ', num2str(runtime), ' seconds']);

% Visualize the final enhanced image after GWO
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(gwo_enhanced_img, []);
title('GWO Enhanced Image (RGB)');

% Step 3: Evaluate the enhancement using the valuation function
disp('Evaluating the enhancement...');
[psnr_value, ssim_value, mse_value, snr_value] = valuation(input_img, gwo_enhanced_img);

% Display evaluation metrics
disp('Evaluation Metrics:');
disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);
disp(['MSE: ', num2str(mse_value)]);
disp(['SNR: ', num2str(snr_value)]);

disp('GWO enhancement process completed.');
