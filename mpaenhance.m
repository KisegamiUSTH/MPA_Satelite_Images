function enhanced_img = mpaenhance(input_img)
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
    upperBound = 5.0;

    % Initialize MPA variables
    bestSolution = [];
    bestFitness = Inf;
    prevBestFitness = Inf;

    % Main MPA loop
    for generation = 1:numGenerations
        % Generate random solutions for beta parameters
        population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);  

        % Evaluate the fitness of each solution
        fitness = evaluateFitness(population, clahe_img_gpu, bilateral_img_gpu, unsharp_img_gpu, gamma_img_gpu, input_img_gpu);

        % Find the best solution and its fitness
        [currentBestFitness, bestIndex] = min(fitness);

        % Update the best solution if a better one is found
        if currentBestFitness < bestFitness
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
        end

        % Check if mutation is needed (if fitness stagnates)
        if generation > 5 && abs(prevBestFitness - bestFitness) < 1e-6
            mutationRate = 0.1;  % Introduce mutation in 10% of the population
            for j = 1:round(mutationRate * populationSize)
                idx = randi([1 populationSize]);
                population(idx, :) = lowerBound + (upperBound - lowerBound) * rand(1, 4); 
            end
            disp('Mutation applied to the population.');
        end

        prevBestFitness = bestFitness;  % Store current best fitness for comparison

        % Display progress after each generation
        disp(['Generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
    end

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

function fitness = evaluateFitness(population, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img)
    numSolutions = size(population, 1);  % Number of solutions
    fitness = zeros(numSolutions, 1);  % Initialize fitness
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

        % Display the calculated values
        %disp(['Solution ', num2str(i)]);
        %disp(['  Variance (V): ', num2str(V)]);
        %disp(['  Entropy of original image (E_1): ', num2str(E_1)]);
        %disp(['  Entropy of enhanced image (E_2): ', num2str(E_2)]);
        %disp(['  Mean Absolute Deviation of original image (G_1): ', num2str(G_1)]);
        %disp(['  Mean Absolute Deviation of enhanced image (G_2): ', num2str(G_2)]);
        %disp(['  PSNR: ', num2str(PSNR)]);
        %disp(['  Penalty: ', num2str(penalty)]);

        fitness(i) = (V / M) * ((E_1 - E_2) + ((G_1 - G_2) / PSNR)) + 0.001 * penalty;
    end
end
