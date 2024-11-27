clear;
clc;

% GPU Initialization
gpuDeviceCount;
dg = gpuDevice;
disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('6100_0_2.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Step 1: Apply Enhancements using the predefined methods
disp('Applying enhancements...');
enhanced_images = enhancements(input_img);

% List of enhancement methods
methods = {'clahe', 'egif', 'nlm', 'histogram', 'unsharp'};

% Initialize results as an empty cell array
results = cell(0, 7);  % 7 columns: Method, MSE, PSNR, SSIM, SNR, Original Entropy, Enhanced Entropy

% Evaluate each enhancement method
for i = 1:length(methods)
    method_name = methods{i};
    enhanced_img = enhanced_images.(method_name);
    
    % Evaluate using the updated valuation function
    [psnr_value, ssim_value, mse_value, snr_value, original_entropy, enhanced_entropy] = valuation(input_img, enhanced_img);
    
    % Store the results
    results = [results; {method_name, mse_value, psnr_value, ssim_value, snr_value, original_entropy, enhanced_entropy}];
end

% Step 2: Apply MPA Optimization
disp('Running MPA optimization on GPU...');
mpa_enhanced_img = mpaenhance(input_img);

% Save the MPA enhanced image as a .jpg file
output_file_name = 'mpa_enhanced_image.jpg';
imwrite(mpa_enhanced_img, output_file_name);
disp(['MPA Enhanced image saved as: ', output_file_name]);

% Step 3: Display images with the requested layout
figure;

% First row: Original image in the middle
subplot(3, 3, 2);  % Middle position of the 1st row
imshow(input_img, []);
title('Original Image (RGB)');

% Second row: MPA Enhanced image first, followed by two other enhancements
subplot(3, 3, 4);
imshow(mpa_enhanced_img, []);
title('MPA Enhanced Image');

subplot(3, 3, 5);
imshow(enhanced_images.clahe, []);
title('CLAHE Enhanced Image');

subplot(3, 3, 6);
imshow(enhanced_images.egif, []);
title('EGIF Enhanced Image');

% Third row: Remaining enhanced images
subplot(3, 3, 7);
imshow(enhanced_images.nlm, []);
title('NLM Enhanced Image');

subplot(3, 3, 8);
imshow(enhanced_images.histogram, []);
title('Histogram Equalized Image');

subplot(3, 3, 9);
imshow(enhanced_images.unsharp, []);
title('Unsharp Masked Image');

disp('Enhancement process completed.');
