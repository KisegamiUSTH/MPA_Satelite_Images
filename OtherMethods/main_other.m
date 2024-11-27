clear;
clc;

% GPU Initialization (Optional)
%gpuDeviceCount;
%dg = gpuDevice;
%disp(['Using GPU: ', dg.Name]);

% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('6100_0_2.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to the range [0, 1]
input_img = im2double(mat2gray(input_img));

% Apply Enhancements using the defined methods
disp('Applying enhancements...');
enhanced_images = enhancements(input_img);

% List of enhancement methods
methods = {'clahe', 'egif', 'nlm', 'histogram', 'unsharp'};

% Initialize results as an empty cell array with the correct number of columns
results = cell(0, 10);  % 10 columns for: Method, MSE, PSNR, SSIM, SNR, Original Entropy, Enhanced Entropy, MLI, AG, CI

% Evaluate each enhancement method
for i = 1:length(methods)
    method_name = methods{i};
    enhanced_img = enhanced_images.(method_name);
    
    % Evaluate using the updated valuation function
    [psnr_value, ssim_value, mse_value, snr_value, original_entropy, enhanced_entropy, mli_value, ag_value, ci_value] = valuation(input_img, enhanced_img);
    
    % Store the results: Method, MSE, PSNR, SSIM, SNR, Original Entropy, Enhanced Entropy, MLI, AG, CI
    results = [results; {method_name, mse_value, psnr_value, ssim_value, snr_value, original_entropy, enhanced_entropy, mli_value, ag_value, ci_value}];
    
    % Display the results
    disp(['Results for ', upper(method_name), ':']);
    fprintf('MSE: %.4f\n', mse_value);
    fprintf('PSNR: %.4f dB\n', psnr_value);
    fprintf('SSIM: %.4f\n', ssim_value);
    fprintf('SNR: %.4f dB\n', snr_value);
    fprintf('Original Entropy: %.4f\n', original_entropy);
    fprintf('Enhanced Entropy: %.4f\n', enhanced_entropy);
    fprintf('MLI: %.4f\n', mli_value);
    fprintf('AG: %.4f\n', ag_value);
    fprintf('CI: %.4f\n', ci_value);
    fprintf('------------------------\n');
    
    % Visualize the enhanced image
    figure;
    subplot(1, 2, 1);
    imshow(input_img, []);
    title('Original Image (RGB)');
    subplot(1, 2, 2);
    imshow(enhanced_img, []);
    title(['Enhanced Image (', upper(method_name), ')']);
end

% Convert results to a table for easy visualization
results_table = cell2table(results, 'VariableNames', {'Method', 'MSE', 'PSNR', 'SSIM', 'SNR', 'Original Entropy', 'Enhanced Entropy', 'MLI', 'AG', 'CI'});
disp(results_table);

% Save the results table as CSV for further analysis
writetable(results_table, 'enhancement_comparison_results.csv');

% Display message
disp('Enhancement comparison completed and results saved as CSV.');
