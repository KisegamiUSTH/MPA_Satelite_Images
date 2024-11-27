function [enhanced_img, bestFitness] = da_enhance(input_img)
    input_img = im2double(input_img);
    input_img(isinf(input_img) | isnan(input_img)) = 0;
    input_img = mat2gray(input_img); 

    % Step 1: Apply CLAHE
    disp('Applying CLAHE...');
    clahe_img = input_img;
    for i = 1:size(input_img, 3)
        clahe_img(:,:,i) = adapthisteq(input_img(:,:,i), 'ClipLimit', 0.05, 'NumTiles', [16 16]);
    end

    % Step 2: Apply Bilateral Filtering
    disp('Applying Bilateral Filtering...');
    bilateral_img = input_img;
    for i = 1:size(input_img, 3)
        bilateral_img(:,:,i) = imbilatfilt(input_img(:,:,i), 0.05, 2);  
    end

    % Step 3: Apply Unsharp Masking
    disp('Applying Unsharp Masking...');
    unsharp_img = input_img;
    for i = 1:size(input_img, 3)
        unsharp_img(:,:,i) = imsharpen(input_img(:,:,i), 'Radius', 2.0, 'Amount', 1.5); 
    end

    % Step 4: Apply Gamma Correction
    disp('Applying Gamma Correction...');
    gamma_value = 1.1;  
    gamma_img = input_img;
    for i = 1:size(input_img, 3)
        gamma_img(:,:,i) = imadjust(input_img(:,:,i), [], [], gamma_value);
    end

    % DA parameters
    populationSize = 150;
    numGenerations = 30;
    lowerBound = 0;
    upperBound = 1.5;
    delta = 0.1; % Step size
    w = 0.9; % Initial inertia weight for more exploration
    mutationRate = 0.2; % Mutation rate
    mutationStrength = 0.1; % Strength of mutation perturbation

    % Initialize population (position and velocity)
    population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);
    velocity = zeros(populationSize, 4);

    % Best solution variables
    bestFitness = Inf;
    bestSolution = [];
    bestMetrics = struct();

    % Main DA loop
    for generation = 1:numGenerations
        % Evaluate fitness of each dragonfly
        [fitness, metrics] = evaluateFitness(population, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img);
        
        % Update the best solution and fitness
        [currentBestFitness, bestIndex] = min(fitness);
        
        % Store the best fitness and metrics
        if currentBestFitness < bestFitness
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
            bestMetrics = metrics(bestIndex);
        end

        % Display progress
        disp(['Generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
    end

    % Apply the optimal parameters to enhance the image
    beta_1 = bestSolution(1);
    beta_2 = bestSolution(2);
    beta_3 = bestSolution(3);
    beta_4 = bestSolution(4);
    enhanced_img = beta_1 * clahe_img + beta_2 * bilateral_img + beta_3 * unsharp_img + beta_4 * gamma_img;

    % Clamp the final enhanced image to [0, 1] range
    enhanced_img = min(max(enhanced_img, 0), 1);

    % Display the best metrics
    disp('Best Metrics after Optimization:');
    disp(['E1 (Original Entropy): ', num2str(bestMetrics.E_1)]);
    disp(['E2 (Enhanced Entropy): ', num2str(bestMetrics.E_2)]);
    disp(['G1 (Original Mean Abs. Dev.): ', num2str(bestMetrics.G_1)]);
    disp(['G2 (Enhanced Mean Abs. Dev.): ', num2str(bestMetrics.G_2)]);
    disp(['PSNR: ', num2str(bestMetrics.PSNR)]);
    disp(['Penalty: ', num2str(bestMetrics.penalty)]);
end



function [fitness, metrics] = evaluateFitness(population, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img)
    numSolutions = size(population, 1);  % Number of solutions
    fitness = zeros(numSolutions, 1);  % Initialize fitness
    metrics = struct('V', [], 'E_1', [], 'E_2', [], 'G_1', [], 'G_2', [], 'PSNR', [], 'penalty', []);
    M = max(input_img(:));  % Maximum pixel value across all bands

    for i = 1:numSolutions
        % Get the combination weights (beta values) from the population
        beta_1 = population(i, 1);
        beta_2 = population(i, 2);
        beta_3 = population(i, 3);
        beta_4 = population(i, 4);

        % Combine the enhanced images with the current weights
        I_T = beta_1 * clahe_img + beta_2 * bilateral_img + beta_3 * unsharp_img + beta_4 * gamma_img;

        % Calculate fitness function components (scalars)
        V = var(I_T(:));  % Variance of the enhanced image
        E_1 = entropy(input_img);  % Entropy of the original image
        E_2 = entropy(I_T);  % Entropy of the enhanced image
        G_1 = mean(abs(input_img(:) - mean(input_img(:))));  % Mean absolute deviation of the original image
        G_2 = mean(abs(I_T(:) - mean(I_T(:))));  % Mean absolute deviation of the enhanced image
        PSNR = 10 * log10(M^2 / mean((I_T(:) - input_img(:)).^2));  % Peak Signal-to-Noise Ratio (scalar)

        % Ensure that PSNR is not zero to avoid division by zero
        if PSNR == 0
            PSNR = 1e-10;
        end

        % Introduce penalty for pixels that exceed the [0, 1] range
        penalty = sum(I_T(:) > 1 | I_T(:) < 0);

        % Store metrics
        metrics(i).V = V;
        metrics(i).E_1 = E_1;
        metrics(i).E_2 = E_2;
        metrics(i).G_1 = G_1;
        metrics(i).G_2 = G_2;
        metrics(i).PSNR = PSNR;
        metrics(i).penalty = penalty;

        % Fitness function calculation
        fitness(i) = (V / M) * ((E_1 - E_2) + ((G_1 - G_2) / PSNR)) + 0.001 * penalty;
    end
end
