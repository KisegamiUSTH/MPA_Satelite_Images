function [enhanced_img, bestFitness] = sca_enhance(input_img) 
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

    % SCA parameters
    populationSize = 100;
    numGenerations = 30;
    lowerBound = 0;
    upperBound = 2.0;

    % Initialize population
    positions = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);

    % Initialize best solution
    bestFitness = inf;
    bestSolution = [];
    bestMetrics = []; % Variable to store the best metrics

    % Main SCA loop
    for generation = 1:numGenerations
        % Evaluate fitness of each solution
        [fitness, metrics] = evaluateFitness(positions, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img);

        % Update the best solution
        for i = 1:populationSize
            if fitness(i) < bestFitness
                bestFitness = fitness(i);
                bestSolution = positions(i, :);
                bestMetrics = metrics(i); % Store the best metrics
            end
        end

        % Early stopping condition
        if bestFitness <= -1
            disp(['Early stopping at generation: ', num2str(generation), ' | Best Fitness: ', num2str(bestFitness)]);
            break;
        end

        % Update positions using the SCA formula
        a = 2 - generation * (2 / numGenerations); % Linearly decreasing parameter a
        for i = 1:populationSize
            for j = 1:4
                r1 = rand(); r2 = rand(); r3 = rand(); r4 = rand();
                if r4 < 0.5
                    positions(i, j) = positions(i, j) + a * sin(r1) * abs(r2 * bestSolution(j) - positions(i, j));
                else
                    positions(i, j) = positions(i, j) + a * cos(r1) * abs(r2 * bestSolution(j) - positions(i, j));
                end
            end

            % Ensure positions remain within bounds
            positions(i, :) = max(min(positions(i, :), upperBound), lowerBound);
        end

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
