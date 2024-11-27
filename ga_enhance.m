function [enhanced_img, bestFitness] = ga_enhance(input_img)
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

    % GA parameters
    populationSize = 100;
    numGenerations = 20;
    mutationRate = 0.1;
    crossoverRate = 0.7;
    lowerBound = 0;
    upperBound = 2.0;

    % Initialize population
    population = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);

    % Best solution variables
    bestFitness = Inf;
    bestSolution = [];
    bestMetrics = struct('V', [], 'E_1', [], 'E_2', [], 'G_1', [], 'G_2', [], 'PSNR', [], 'penalty', []);
    lastBestFitness = Inf;  % Variable to store the last best fitness before early stopping

    % Main GA loop
    for generation = 1:numGenerations
        % Evaluate fitness of each individual
        [fitness, metrics] = evaluateFitness(population, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img);
        
        % Find the best solution and its fitness
        [currentBestFitness, bestIndex] = min(fitness);
        if currentBestFitness < bestFitness
            lastBestFitness = bestFitness;  % Store the previous best fitness
            bestFitness = currentBestFitness;
            bestSolution = population(bestIndex, :);
            bestMetrics = metrics(bestIndex); % Store the metrics of the best solution
        end

        % Early stopping condition
        if bestFitness < -1
            disp(['Early stopping at generation: ', num2str(generation), ' | Best Fitness: ', num2str(lastBestFitness)]);
            % Use the last best fitness before this generation
            bestFitness = lastBestFitness;
            break;
        end

        % Selection (Roulette Wheel)
        fitnessInv = 1 ./ (fitness + 1e-6); % Avoid division by zero
        selectionProb = fitnessInv / sum(fitnessInv);

        % Fix for randsample error: ensure selectionProb is non-negative and normalize it
        selectionProb = max(selectionProb, 0);  % Ensure non-negative values
        if sum(selectionProb) == 0
            % If all selectionProb values are zero, set uniform distribution
            selectionProb = ones(1, populationSize) / populationSize;
        else
            % Normalize the selectionProb to ensure they sum to 1
            selectionProb = selectionProb / sum(selectionProb);
        end

        % Select individuals based on selection probabilities
        selectedIndices = randsample(1:populationSize, populationSize, true, selectionProb);
        selectedPopulation = population(selectedIndices, :);

        % Crossover
        for i = 1:2:populationSize-1
            if rand < crossoverRate
                % Single point crossover
                crossoverPoint = randi([1, 3]); % Between 1 and 3 for 4 parameters
                parent1 = selectedPopulation(i, :);
                parent2 = selectedPopulation(i+1, :);
                child1 = [parent1(1:crossoverPoint), parent2(crossoverPoint+1:end)];
                child2 = [parent2(1:crossoverPoint), parent1(crossoverPoint+1:end)];
                selectedPopulation(i, :) = child1;
                selectedPopulation(i+1, :) = child2;
            end
        end

        % Mutation
        for i = 1:populationSize
            if rand < mutationRate
                mutation = 0.5 * (rand(1, 4) - 0.5); % Random small perturbation
                selectedPopulation(i, :) = selectedPopulation(i, :) + mutation;
                % Ensure the mutated individual remains within bounds
                selectedPopulation(i, :) = max(min(selectedPopulation(i, :), upperBound), lowerBound);
            end
        end

        % Update population
        population = selectedPopulation;

        % Display progress
        disp(['Generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
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

    % Apply the optimal parameters to enhance the image
    beta_1 = bestSolution(1);
    beta_2 = bestSolution(2);
    beta_3 = bestSolution(3);
    beta_4 = bestSolution(4);
    enhanced_img = beta_1 * clahe_img + beta_2 * bilateral_img + beta_3 * unsharp_img + beta_4 * gamma_img;

    % Clamp the final enhanced image to [0, 1] range
    enhanced_img = min(max(enhanced_img, 0), 1);
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
