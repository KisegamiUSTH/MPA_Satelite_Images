function enhanced_images = enhancements(input_img)
    % Ensure the input image is in the range [0, 1]
    input_img = im2double(input_img);
    
    % 1. CLAHE
    clahe_img = input_img;
    for i = 1:size(input_img, 3)
        clahe_img(:, :, i) = adapthisteq(input_img(:, :, i));
    end

    % 2. Effective Guided Image Filtering (EGIF)
    egif_img = effective_guided_image_filtering(input_img);

    % 3. Non-Local Means Filtering
    nlm_img = input_img;
    for i = 1:size(input_img, 3)
        nlm_img(:, :, i) = imnlmfilt(input_img(:, :, i));
    end

    % 4. Histogram Matching/Equalization
    hist_img = input_img;
    for i = 1:size(input_img, 3)
        hist_img(:, :, i) = histeq(input_img(:, :, i));
    end
    
    % 5. Unsharp Masking
    unsharp_img = input_img;
    for i = 1:size(input_img, 3)
        unsharp_img(:, :, i) = imsharpen(input_img(:, :, i), 'Radius', 2.0, 'Amount', 1.5);
    end
    
    % Store all enhanced images in a struct
    enhanced_images.clahe = clahe_img;
    enhanced_images.egif = egif_img;
    enhanced_images.nlm = nlm_img;
    enhanced_images.histogram = hist_img;
    enhanced_images.unsharp = unsharp_img;
end

function egif_img = effective_guided_image_filtering(input_img)
    % Dummy implementation for Effective Guided Image Filtering (EGIF)
    % Normally, this would involve guided filtering steps
    egif_img = imguidedfilter(input_img);  % Simplified version
end
