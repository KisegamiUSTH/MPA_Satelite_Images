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

% Number of runs
numRuns = 20;
bestFitnessValues = zeros(numRuns, 1);  % Array to store the best fitness from each run
runtimeValues = zeros(numRuns, 1);  % Array to store the runtime of each run

% Loop for 20 runs
for i = 1:numRuns
    disp(['Running MPA optimization - Run ' num2str(i) '...']);
    
    % Measure the runtime for each run
    tic;
    
    % Apply MPA Enhancement and retrieve the best fitness
    [mpa_enhanced_img, bestFitness] = mpaenhance(input_img);
    bestFitnessValues(i) = bestFitness;  % Store the best fitness
    
    % Store the runtime for each run
    runtimeValues(i) = toc;
    
    % Save the enhanced image for each run
    output_file_name = ['mpa_enhanced_image_run_' num2str(i) '.jpg'];
    imwrite(mpa_enhanced_img, output_file_name);
    disp(['Enhanced image for Run ' num2str(i) ' saved as: ', output_file_name]);
    disp(['Best Fitness for Run ' num2str(i) ': ' num2str(bestFitness)]);
    disp(['Runtime for Run ' num2str(i) ': ' num2str(runtimeValues(i)) ' seconds']);
end

% Exclude Inf values from bestFitnessValues
finite_fitness_values = bestFitnessValues(~isinf(bestFitnessValues));

% Calculate and print overall statistics for finite fitness values
if ~isempty(finite_fitness_values)
    bestFitnessOverall = min(finite_fitness_values);
    worstFitnessOverall = max(finite_fitness_values);
    meanFitness = mean(finite_fitness_values);
    stdFitness = std(finite_fitness_values);
else
    bestFitnessOverall = NaN;
    worstFitnessOverall = NaN;
    meanFitness = NaN;
    stdFitness = NaN;
    disp('No valid fitness values found.');
end

% Display the results
disp('--------------------------------------------');
disp('Summary of 20 Runs (excluding Inf fitness values):');
disp(['Best Fitness Overall: ', num2str(bestFitnessOverall)]);
disp(['Worst Fitness Overall: ', num2str(worstFitnessOverall)]);
disp(['Mean Fitness: ', num2str(meanFitness)]);
disp(['Standard Deviation of Fitness: ', num2str(stdFitness)]);
disp(['Average Runtime per Run: ', num2str(mean(runtimeValues)), ' seconds']);
disp('--------------------------------------------');

% List every best fitness value from the 20 runs
disp('Best Fitness Values for Each Run:');
for i = 1:numRuns
    disp(['Run ' num2str(i) ': Best Fitness = ' num2str(bestFitnessValues(i))]);
end

