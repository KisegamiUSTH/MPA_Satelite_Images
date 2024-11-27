% Load the 3-band multi-spectral satellite image (.tif)
input_img = imread('tile_2_4.tif');  % Replace with your actual image path

% Check the number of bands in the image
[H, W, numBands] = size(input_img);
if numBands ~= 3
    error('This image does not contain 3 bands. Please ensure it is a 3-band multi-spectral image.');
end

% Normalize the image to process properly
input_img = im2double(mat2gray(input_img));  % Normalize to [0, 1] range and convert to double

% Compute AG (Average Gradient) for each band
ag_values = zeros(1, 3);
for i = 1:3
    [Gx, Gy] = gradient(input_img(:,:,i));
    ag_values(i) = mean(mean(sqrt(Gx.^2 + Gy.^2)));
end
avg_ag = mean(ag_values);
fprintf('Average Gradient (AG): %.4f\n', avg_ag);

% Compute MLI (Mean Light Intensity) for each band
mli_values = zeros(1, 3);
for i = 1:3
    % Calculate Mean Light Intensity
    mli_values(i) = mean(mean(input_img(:,:,i)));
end
avg_mli = mean(mli_values);
fprintf('Mean Light Intensity (MLI): %.4f\n', avg_mli);
