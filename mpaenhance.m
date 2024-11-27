function [enhanced_img, bestFitness] = mpaenhance(input_img)
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

    % Initialize GPU for MPA
    ag = gpuDeviceCount;
    if ag > 0
        % Move the enhanced images to the GPU for MPA optimization
        clahe_img_gpu = gpuArray(clahe_img);
        bilateral_img_gpu = gpuArray(bilateral_img);
        unsharp_img_gpu = gpuArray(unsharp_img);
        gamma_img_gpu = gpuArray(gamma_img);
        input_img_gpu = gpuArray(input_img);
    else
        clahe_img_gpu = clahe_img;
        bilateral_img_gpu = bilateral_img;
        unsharp_img_gpu = unsharp_img;
        gamma_img_gpu = gamma_img;
        input_img_gpu = input_img;
    end

    % MPA parameters
    populationSize = 150;
    numGenerations = 50;
    lowerBound = 0; 
    upperBound = 1.5;

    % Initialize MPA variables
    bestSolution = [];
    bestFitness = Inf;
    lastPositiveFitness = Inf;  % Variable to store the last positive best fitness
    prevBestFitness = Inf;

    % Variable to store metrics of the best fitness
    bestMetrics = struct('V', [], 'E_1', [], 'E_2', [], 'G_1', [], 'G_2', [], 'PSNR', [], 'penalty', []);

    % Main MPA loop
    for generation = 1:numGenerations
        % Generate random solutions for beta parameters
        population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);  

        % Evaluate the fitness of each solution
        [fitness, metrics] = evaluateFitness(population, clahe_img_gpu, bilateral_img_gpu, unsharp_img_gpu, gamma_img_gpu, input_img_gpu);

        % Find the best solution and its fitness
        [currentBestFitness, bestIndex] = min(fitness);

        % Update the best solution and metrics if a better one is found
        if currentBestFitness < bestFitness
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
            bestMetrics = metrics(bestIndex);
        end


        % Early stopping condition
        if bestFitness <= -1
            disp(['Early stopping at generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
            % Use the last positive fitness instead of the current bestFitness
            bestFitness = lastPositiveFitness;
            break;
        end

        % Check if mutation is needed (if fitness stagnates)
        if generation > 5 && abs(prevBestFitness - bestFitness) < 1e-6
            mutationRate = 0.1;  % Introduce mutation in 10% of the population
            for j = 1:round(mutationRate * populationSize)
                idx = randi([1 populationSize]);
                population(idx, :) = lowerBound + (upperBound - lowerBound) * rand(1, 4); 
            end
            %disp('Mutation applied to the population.');
        end

        prevBestFitness = bestFitness;  % Store current best fitness for comparison

        % Display progress after each generation
        %disp(['Generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
    end

    % Print the best metrics
    disp('Best fitness metrics:');
    disp(['  Variance (V): ', num2str(bestMetrics.V)]);
    disp(['  Entropy of original image (E_1): ', num2str(bestMetrics.E_1)]);
    disp(['  Entropy of enhanced image (E_2): ', num2str(bestMetrics.E_2)]);
    disp(['  Mean Absolute Deviation of original image (G_1): ', num2str(bestMetrics.G_1)]);
    disp(['  Mean Absolute Deviation of enhanced image (G_2): ', num2str(bestMetrics.G_2)]);
    disp(['  PSNR: ', num2str(bestMetrics.PSNR)]);
    disp(['  Penalty: ', num2str(bestMetrics.penalty)]);

    % Print the best solution
    disp('Best solution (beta values) at best fitness:');
    disp(['  Beta_1 (CLAHE): ', num2str(bestSolution(1))]);
    disp(['  Beta_2 (Bilateral Filtering): ', num2str(bestSolution(2))]);
    disp(['  Beta_3 (Unsharp Masking): ', num2str(bestSolution(3))]);
    disp(['  Beta_4 (Gamma Correction): ', num2str(bestSolution(4))]);

    % Apply the optimal parameters to enhance the image
    beta_1 = bestSolution(1);
    beta_2 = bestSolution(2);
    beta_3 = bestSolution(3);
    beta_4 = bestSolution(4);
    enhanced_img_gpu = beta_1 * clahe_img_gpu + beta_2 * bilateral_img_gpu + beta_3 * unsharp_img_gpu + beta_4 * gamma_img_gpu;

    % Clamp the final enhanced image to [0, 1] range
    enhanced_img_gpu = min(max(enhanced_img_gpu, 0), 1);

    enhanced_img = gather(enhanced_img_gpu);  % Move result back to CPU (if using GPU)
end

function [fitness, metrics] = evaluateFitness(population, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img)
    numSolutions = size(population, 1);  % Number of solutions
    fitness = zeros(numSolutions, 1);  % Initialize fitness
    metrics = struct('V', [], 'E_1', [], 'E_2', [], 'G_1', [], 'G_2', [], 'PSNR', [], 'penalty', []);
    [H, W, C] = size(input_img);  % Image dimensions
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

        fitness(i) = (V / M) * ((E_1 - E_2) + ((G_1 - G_2) / PSNR)) + 0.001 * penalty;
    end
end
