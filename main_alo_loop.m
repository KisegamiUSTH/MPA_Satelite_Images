clear;
clc;

% GPU Initialization
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

% Initialize array to store best fitness values and runtime
fitness_values = zeros(20, 1);
runtime_values = zeros(20, 1);

% Loop 20 times for ALO enhancement
totalStartTime = tic;  % Start overall timer
for i = 1:20
    disp(['Running ALO optimization, iteration: ', num2str(i)]);
    
    % Measure the time for each run
    iterationStartTime = tic;
    
    % Apply ALO Optimization (GPU-Enabled)
    [alo_enhanced_img, bestFitness] = alo_enhance(input_img);
    
    % Store the best fitness value for this run
    fitness_values(i) = bestFitness;
    
    % Store the runtime for this run
    runtime_values(i) = toc(iterationStartTime);
    
    % Display the best fitness for this iteration
    disp(['Best Fitness for iteration ', num2str(i), ': ', num2str(bestFitness)]);
    disp(['Runtime for iteration ', num2str(i), ': ', num2str(runtime_values(i)), ' seconds']);
end
totalTime = toc(totalStartTime);  % Stop overall timer

% Exclude Inf values from fitness_values
finite_fitness_values = fitness_values(~isinf(fitness_values));

% After 20 runs, calculate and display statistics for finite fitness values
if ~isempty(finite_fitness_values)
    best_overall = min(finite_fitness_values);
    worst_overall = max(finite_fitness_values);
    mean_fitness = mean(finite_fitness_values);
    std_dev_fitness = std(finite_fitness_values);
else
    best_overall = NaN;
    worst_overall = NaN;
    mean_fitness = NaN;
    std_dev_fitness = NaN;
    disp('No valid fitness values found.');
end

% Display the results
disp('--------------------');
disp('Summary of 20 runs (excluding Inf fitness values):');
disp(['Best Fitness Overall: ', num2str(best_overall)]);
disp(['Worst Fitness Overall: ', num2str(worst_overall)]);
disp(['Mean Fitness: ', num2str(mean_fitness)]);
disp(['Standard Deviation: ', num2str(std_dev_fitness)]);
disp(['Total Runtime for 20 iterations: ', num2str(totalTime), ' seconds']);
disp(['Average Runtime per iteration: ', num2str(mean(runtime_values)), ' seconds']);

% Save the enhanced image from the last iteration as a .jpg file
output_file_name = 'alo_enhanced_image.jpg';
imwrite(alo_enhanced_img, output_file_name);
disp(['Last enhanced image saved as: ', output_file_name]);

% Visualize the final enhanced image after 20 runs of ALO
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(alo_enhanced_img, []);
title('ALO Enhanced Image (Last Run)');

disp('Enhancement process completed.');

% Step 3: Evaluate the enhancement using the valuation function for the last run
disp('Evaluating the enhancement for the last run...');
[psnr_value, ssim_value, mse_value, snr_value] = valuation(input_img, alo_enhanced_img);

% Display evaluation metrics
disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);
disp(['MSE: ', num2str(mse_value)]);
disp(['SNR: ', num2str(snr_value)]);
