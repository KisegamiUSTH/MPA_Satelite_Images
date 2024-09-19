function [enhanced_img, bestFitness] = gwo_enhance(input_img)
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

    % GWO parameters
    populationSize = 30;
    numGenerations = 20;
    lowerBound = 0;
    upperBound = 5.0;

    % Initialize wolf positions (population)
    wolves = lowerBound + (upperBound - lowerBound) * rand(populationSize, 4);

    % Initialize alpha, beta, and delta wolves
    alpha = inf(1, 4);
    beta = inf(1, 4);
    delta = inf(1, 4);
    alphaFitness = inf;
    betaFitness = inf;
    deltaFitness = inf;
    lastBestFitness = inf;

    % Main GWO loop
    for generation = 1:numGenerations
        % Evaluate the fitness of wolves
        [fitness, metrics] = evaluateFitness(wolves, clahe_img, bilateral_img, unsharp_img, gamma_img, input_img);
        
        % Update alpha, beta, and delta
        for i = 1:populationSize
            if fitness(i) < alphaFitness
                alphaFitness = fitness(i);
                alpha = wolves(i, :);
            elseif fitness(i) < betaFitness
                betaFitness = fitness(i);
                beta = wolves(i, :);
            elseif fitness(i) < deltaFitness
                deltaFitness = fitness(i);
                delta = wolves(i, :);
            end
        end
        
        % Store the last best fitness
        if alphaFitness > 0
            lastBestFitness = alphaFitness;
        end

        % Early stopping condition
        if alphaFitness <= 0
            disp(['Early stopping at generation: ', num2str(generation), ' | Best Fitness: ', num2str(alphaFitness)]);
            % Use the last positive fitness instead of the current bestFitness
            alphaFitness = lastBestFitness;
            break;
        end

        % Update the positions of wolves
        a = 2 - generation * (2 / numGenerations);  % Linearly decreasing coefficient
        for i = 1:populationSize
            for j = 1:4  % For each dimension
                % Calculate the coefficients
                A1 = 2 * a * rand - a;
                C1 = 2 * rand;
                A2 = 2 * a * rand - a;
                C2 = 2 * rand;
                A3 = 2 * a * rand - a;
                C3 = 2 * rand;
                
                % Update wolf position
                D_alpha = abs(C1 * alpha(j) - wolves(i, j));
                D_beta = abs(C2 * beta(j) - wolves(i, j));
                D_delta = abs(C3 * delta(j) - wolves(i, j));
                
                X1 = alpha(j) - A1 * D_alpha;
                X2 = beta(j) - A2 * D_beta;
                X3 = delta(j) - A3 * D_delta;
                
                % New position
                wolves(i, j) = (X1 + X2 + X3) / 3;
            end
            % Ensure the wolves remain within bounds
            wolves(i, :) = max(min(wolves(i, :), upperBound), lowerBound);
        end
        
        % Display progress
        disp(['Generation: ', num2str(generation), ' | Alpha Fitness: ', num2str(alphaFitness)]);
    end

    bestFitness = alphaFitness;  % Final best fitness (alpha wolf's fitness)
    bestSolution = alpha;  % Best solution (alpha wolf's position)

    % Print the best metrics
    disp('Best fitness metrics:');
    disp(['  Variance (V): ', num2str(metrics(1).V)]);
    disp(['  Entropy of original image (E_1): ', num2str(metrics(1).E_1)]);
    disp(['  Entropy of enhanced image (E_2): ', num2str(metrics(1).E_2)]);
    disp(['  Mean Absolute Deviation of original image (G_1): ', num2str(metrics(1).G_1)]);
    disp(['  Mean Absolute Deviation of enhanced image (G_2): ', num2str(metrics(1).G_2)]);
    disp(['  PSNR: ', num2str(metrics(1).PSNR)]);
    disp(['  Penalty: ', num2str(metrics(1).penalty)]);

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
