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

% Variables to store fitness values and runtime
numRuns = 20;
fitness_values = zeros(numRuns, 1);
runtime_values = zeros(numRuns, 1);

% Loop to run SSA enhancement 20 times
totalStartTime = tic;  % Start overall timer
for i = 1:numRuns
    disp(['Running SSA optimization iteration: ', num2str(i)]);
    
    % Measure the time for each run
    iterationStartTime = tic;
    
    % Apply SSA Optimization (GPU-Enabled)
    [ssa_enhanced_img, bestFitness] = ssa_enhance(input_img);
    
    % Record the runtime for this iteration
    iterationTime = toc(iterationStartTime);
    runtime_values(i) = iterationTime;
    
    % Store the best fitness of this run
    fitness_values(i) = bestFitness;
    disp(['Iteration ', num2str(i), ' | Best Fitness: ', num2str(bestFitness), ' | Runtime: ', num2str(iterationTime), ' seconds']);
end
totalTime = toc(totalStartTime);  % Stop overall timer

% Display all best fitness values after the 20 runs
disp('Best Fitness values for each iteration:');
for i = 1:numRuns
    disp(['Run ', num2str(i), ' | Best Fitness: ', num2str(fitness_values(i)), ' | Runtime: ', num2str(runtime_values(i)), ' seconds']);
end

% Calculate the best, worst, mean, and standard deviation of the fitness values (excluding values > 100)
filtered_fitness_values = fitness_values(fitness_values <= 100);
bestFitnessOverall = min(filtered_fitness_values);
worstFitnessOverall = max(filtered_fitness_values);
meanFitness = mean(filtered_fitness_values);
stdFitness = std(filtered_fitness_values);

% Display the overall results
disp('Results after 20 iterations of SSA optimization:');
disp(['Best Fitness Overall: ', num2str(bestFitnessOverall)]);
disp(['Worst Fitness Overall: ', num2str(worstFitnessOverall)]);
disp(['Mean Fitness: ', num2str(meanFitness)]);
disp(['Standard Deviation of Fitness: ', num2str(stdFitness)]);
disp(['Total Runtime for 20 iterations: ', num2str(totalTime), ' seconds']);
disp(['Average Runtime per iteration: ', num2str(mean(runtime_values)), ' seconds']);
