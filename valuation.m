function [psnr_value, ssim_value, mse_value, snr_value] = valuation(original_image, enhanced_image)
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

    % Display the results
    fprintf('MSE: %.4f\n', mse_value);
    fprintf('PSNR: %.4f dB\n', psnr_value);
    fprintf('SSIM: %.4f\n', ssim_value);
    fprintf('SNR: %.4f dB\n', snr_value);
end
