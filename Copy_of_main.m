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

% Loop for 30 runs
for i = 1:numRuns
    disp(['Running MPA optimization on GPU - Run ' num2str(i) '...']);
    
    % Apply MPA Enhancement and retrieve the best fitness
    [mpa_enhanced_img, bestFitness] = mpaenhance(input_img);
    bestFitnessValues(i) = bestFitness;  % Store the best fitness
    
    % Save the enhanced image for each run (optional)
    output_file_name = ['mpa_enhanced_image_run_' num2str(i) '.jpg'];
    imwrite(mpa_enhanced_img, output_file_name);
    disp(['Enhanced image for Run ' num2str(i) ' saved as: ', output_file_name]);
end

% Calculate and print overall statistics
bestFitnessOverall = min(bestFitnessValues);
worstFitnessOverall = max(bestFitnessValues);
meanFitness = mean(bestFitnessValues);
stdFitness = std(bestFitnessValues);

disp('--------------------------------------------');
disp('Summary of 30 Runs:');
disp(['Best Fitness Overall: ', num2str(bestFitnessOverall)]);
disp(['Worst Fitness Overall: ', num2str(worstFitnessOverall)]);
disp(['Mean Fitness: ', num2str(meanFitness)]);
disp(['Standard Deviation of Fitness: ', num2str(stdFitness)]);
disp('--------------------------------------------');

