function applyAllEnhancements(input_img)
    % Ensure the input image is in the correct format
    input_img = im2double(input_img);
    input_img(isinf(input_img) | isnan(input_img)) = 0;
    input_img = mat2gray(input_img); 

    % Step 1: Apply CLAHE
    clahe_img = input_img;
    for i = 1:size(input_img, 3)
        clahe_img(:,:,i) = adapthisteq(input_img(:,:,i));
    end
    imwrite(clahe_img, 'output_clahe.png');
    disp('CLAHE applied and saved as output_clahe.png');

    % Step 2: Apply Bilateral Filtering
    bilateral_img = input_img;
    for i = 1:size(input_img, 3)
        bilateral_img(:,:,i) = imbilatfilt(input_img(:,:,i));
    end
    imwrite(bilateral_img, 'output_bilateral.png');
    disp('Bilateral Filtering applied and saved as output_bilateral.png');

    % Step 3: Apply Unsharp Masking
    unsharp_img = input_img;
    for i = 1:size(input_img, 3)
        unsharp_img(:,:,i) = imsharpen(input_img(:,:,i));
    end
    imwrite(unsharp_img, 'output_unsharp.png');
    disp('Unsharp Masking applied and saved as output_unsharp.png');

    % Step 4: Apply Gamma Correction
    gamma_img = input_img;
    for i = 1:size(input_img, 3)
        gamma_img(:,:,i) = imadjust(input_img(:,:,i));
    end
    imwrite(gamma_img, 'output_gamma.png');
    disp('Gamma Correction applied and saved as output_gamma.png');
end
