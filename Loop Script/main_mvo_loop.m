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

% Number of runs
numRuns = 20;
bestFitnessValues = zeros(numRuns, 1);  % Array to store the best fitness from each run
runTimes = zeros(numRuns, 1);  % Array to store the runtime for each run

% Loop for 20 runs
for i = 1:numRuns
    disp(['Running MVO optimization on GPU - Run ' num2str(i) '...']);
    
    % Measure the start time
    tic;
    
    % Apply MVO Enhancement and retrieve the best fitness
    [mvo_enhanced_img, bestFitness] = mvo_enhance(input_img);
    
    % Measure the runtime for this run
    runTime = toc;
    runTimes(i) = runTime;
    
    bestFitnessValues(i) = bestFitness;  % Store the best fitness
    
    % Save the enhanced image for each run (optional)
    output_file_name = ['mvo_enhanced_image_run_' num2str(i) '.jpg'];
    imwrite(mvo_enhanced_img, output_file_name);
    disp(['Enhanced image for Run ' num2str(i) ' saved as: ', output_file_name]);
    disp(['Best Fitness for Run ' num2str(i) ': ' num2str(bestFitness) ' | Runtime: ' num2str(runTime) ' seconds']);
end

% Exclude Inf values from fitness_values
finite_fitness_values = bestFitnessValues(~isinf(bestFitnessValues));

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

% Calculate average runtime
mean_runtime = mean(runTimes);

% Display the results
disp('--------------------------------------------');
disp('Summary of 20 Runs (excluding Inf fitness values):');
disp(['Best Fitness Overall: ', num2str(best_overall)]);
disp(['Worst Fitness Overall: ', num2str(worst_overall)]);
disp(['Mean Fitness: ', num2str(mean_fitness)]);
disp(['Standard Deviation of Fitness: ', num2str(std_dev_fitness)]);
disp(['Average Runtime per Run: ', num2str(mean_runtime) ' seconds']);
disp('--------------------------------------------');

% Display best fitness value from each run
disp('Best Fitness Values for Each Run:');
for i = 1:numRuns
    disp(['Run ' num2str(i) ': Best Fitness = ' num2str(bestFitnessValues(i))]);
end

% Visualize the final enhanced image from the last run
figure;
subplot(1, 2, 1);
imshow(input_img, []);
title('Original Image (RGB)');
subplot(1, 2, 2);
imshow(mvo_enhanced_img, []);
title('MVO Enhanced Image (Last Run)');

disp('Enhancement process completed.');

% Step 3: Evaluate the enhancement using the valuation function for the last run
disp('Evaluating the enhancement for the last run...');
[psnr_value, ssim_value, mse_value, snr_value] = valuation(input_img, mvo_enhanced_img);

% Display evaluation metrics
disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);
disp(['MSE: ', num2str(mse_value)]);
disp(['SNR: ', num2str(snr_value)]);
