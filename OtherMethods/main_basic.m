clear;
clc;

% GPU Initialization
%gpuDeviceCount;
%dg = gpuDevice;
%disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('6100_0_3.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Apply all basic enhancements and save the results
disp('Applying basic enhancements...');
applyAllEnhancements(input_img);
disp('All basic enhancements applied and images saved.');

% Visualize the original image and the enhanced images
figure;
subplot(2, 3, 1);
imshow(input_img, []);
%title('Original Image (RGB)');

% Load and display the enhanced images
clahe_img = imread('output_clahe.png');
bilateral_img = imread('output_bilateral.png');
unsharp_img = imread('output_unsharp.png');
gamma_img = imread('output_gamma.png');

subplot(2, 3, 2);
imshow(clahe_img, []);
%title('CLAHE Enhanced Image');

subplot(2, 3, 3);
imshow(bilateral_img, []);
%title('Bilateral Filtering Enhanced Image');

subplot(2, 3, 4);
imshow(unsharp_img, []);
%title('Unsharp Masking Enhanced Image');

subplot(2, 3, 5);
imshow(gamma_img, []);
%title('Gamma Correction Enhanced Image');

disp('Enhancement process completed.');
