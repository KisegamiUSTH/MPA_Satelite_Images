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

% Initialize variables for looping
numRuns = 20;
bestFitnessValues = zeros(numRuns, 1);  % Array to store the best fitness from each run
runtimeValues = zeros(numRuns, 1);  % Array to store runtime for each run
validRuns = 0;  % Counter for valid runs

% Loop for 20 runs
for i = 1:numRuns
    disp(['Running DA optimization on GPU - Run ' num2str(i) '...']);
    
    % Start timing for this run
    tic;
    
    try
        % Apply DA Optimization and retrieve the best fitness
        [da_enhanced_img, bestFitness] = da_enhance(input_img);  % Ensure da_enhance returns best fitness
        
        % Record runtime for this run
        runtimeValues(i) = toc;
        
        % Skip runs with Inf fitness
        if ~isfinite(bestFitness)
            disp(['Run ' num2str(i) ' skipped due to Inf fitness']);
            continue;  % Skip this run
        end
        
        % Save the enhanced image for each run (optional)
        output_file_name = ['da_enhanced_image_run_' num2str(i) '.jpg'];
        imwrite(da_enhanced_img, output_file_name);
        disp(['Enhanced image for Run ' num2str(i) ' saved as: ', output_file_name]);
        
        % Store the best fitness (instead of PSNR)
        bestFitnessValues(i) = bestFitness;

        % Display the best fitness value for this run
        disp(['Run ' num2str(i) ' | Best Fitness: ', num2str(bestFitness)]);
        
        validRuns = validRuns + 1;  % Count this run as valid
    catch
        disp(['Run ' num2str(i) ' skipped due to error or Inf fitness']);
        runtimeValues(i) = NaN;  % Mark the runtime as invalid
    end
end

% Exclude Inf and NaN values from fitness_values
finite_fitness_values = bestFitnessValues(isfinite(bestFitnessValues));

% After 20 runs, calculate and display statistics for finite fitness values
if ~isempty(finite_fitness_values)
    bestOverallFitness = min(finite_fitness_values);
    worstOverallFitness = max(finite_fitness_values);
    meanFitness = mean(finite_fitness_values);
    stdDevFitness = std(finite_fitness_values);
else
    bestOverallFitness = NaN;
    worstOverallFitness = NaN;
    meanFitness = NaN;
    stdDevFitness = NaN;
    disp('No valid fitness values found.');
end

% Calculate overall runtime statistics for valid runs
valid_runtime_values = runtimeValues(~isnan(runtimeValues));
totalRuntime = sum(valid_runtime_values);
averageRuntime = mean(valid_runtime_values);

% Display the final statistics
disp('--------------------------------------------');
disp(['Valid Runs: ', num2str(validRuns)]);
disp('Summary of Valid Runs (excluding Inf fitness values):');
disp(['Best Fitness Overall: ', num2str(bestOverallFitness)]);
disp(['Worst Fitness Overall: ', num2str(worstOverallFitness)]);
disp(['Mean Fitness: ', num2str(meanFitness)]);
disp(['Standard Deviation of Fitness: ', num2str(stdDevFitness)]);
disp(['Total Runtime for valid runs: ', num2str(totalRuntime), ' seconds']);
disp(['Average Runtime per Run: ', num2str(averageRuntime), ' seconds']);
disp('--------------------------------------------');

% Display the best fitness for each run
disp('Best Fitness Values for Each Valid Run:');
for i = 1:numRuns
    if isfinite(bestFitnessValues(i))
        disp(['Run ' num2str(i) ': Best Fitness = ', num2str(bestFitnessValues(i)), ' | Runtime: ', num2str(runtimeValues(i)), ' seconds']);
    end
end

disp('DA optimization process completed.');
