function [psnr_value, ssim_value, mse_value, snr_value, original_entropy, enhanced_entropy, mli_value, ag_value, ci_value] = valuation(original_image, enhanced_image)
    % Convert images to double for calculation
    original_image = double(original_image);
    enhanced_image = double(enhanced_image);
    
    % Calculate Mean Squared Error (MSE)
    mse_value = immse(enhanced_image, original_image);

    % Calculate Peak Signal-to-Noise Ratio (PSNR)
    psnr_value = psnr(enhanced_image, original_image);

    % Calculate Structural Similarity Index (SSIM)
    ssim_value = ssim(enhanced_image, original_image);

    % Calculate Signal-to-Noise Ratio (SNR)
    signal_power = mean(original_image(:).^2);
    noise_power = mean((original_image(:) - enhanced_image(:)).^2);
    snr_value = 10 * log10(signal_power / noise_power);

    % Calculate Entropy for both original and enhanced images
    original_entropy = entropy(original_image / max(original_image(:)));
    enhanced_entropy = entropy(enhanced_image / max(enhanced_image(:)));
    
    % Calculate Mean Local Intensity (MLI)
    mli_value = mean2(enhanced_image);

    % Calculate Average Gradient (AG)
    [gx, gy] = gradient(enhanced_image);
    ag_value = mean2(sqrt(gx.^2 + gy.^2));

    % Calculate Contrast Improvement (CI)
    original_contrast = max(original_image(:)) - min(original_image(:));
    enhanced_contrast = max(enhanced_image(:)) - min(enhanced_image(:));
    ci_value = (enhanced_contrast - original_contrast) / original_contrast;

    % Display the results
    fprintf('MSE: %.4f\n', mse_value);
    fprintf('PSNR: %.4f dB\n', psnr_value);
    fprintf('SSIM: %.4f\n', ssim_value);
    fprintf('SNR: %.4f dB\n', snr_value);
    fprintf('Original Entropy: %.4f\n', original_entropy);
    fprintf('Enhanced Entropy: %.4f\n', enhanced_entropy);
    fprintf('MLI: %.4f\n', mli_value);
    fprintf('AG: %.4f\n', ag_value);
    fprintf('CI: %.4f\n', ci_value);
end
