# task3.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from task2 import add_salt_and_pepper_noise, add_gaussian_noise, add_rayleigh_noise, alpha_trimmed_mean_filter
import pandas as pd

# Function to apply a weighted median filter to an image
def weighted_median_filter(image, weights):
    # Pad the image to handle borders using 'edge' mode to replicate the border values
    padded_image = np.pad(image, (1, 1), mode='edge')
    # Create an output image with the same shape as the input image to store the filtered results
    output_image = np.zeros_like(image)

    # Loop over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the 3x3 kernel around the current pixel
            kernel = padded_image[i:i + 3, j:j + 3]
            # Flatten the kernel into a 1D array and repeat each value according to the weights
            weighted_values = np.repeat(kernel.flatten(), weights.flatten())
            # Sort the weighted values to prepare for finding the median
            sorted_values = np.sort(weighted_values)
            # Find the median of the sorted weighted values
            median_value = np.median(sorted_values)
            # Assign the median value to the corresponding pixel in the output image
            output_image[i, j] = median_value

    # Convert the output image to an unsigned 8-bit integer format and return it
    return output_image.astype(np.uint8)

# Function to calculate the Peak Signal-to-Noise Ratio (PSNR) between two images
def compute_psnr(original, denoised):
    # Use the skimage.metrics.peak_signal_noise_ratio function to calculate PSNR
    # with a data range of 255, indicating 8-bit images
    return psnr(original, denoised, data_range=255)

# Weights for the 3x3 weighted median filter kernel
weights = np.array([
    [1, 2, 1],  # Rows represent the weighting of each pixel in the 3x3 neighborhood
    [2, 3, 2],
    [1, 2, 1]
])

# Load the original grayscale image using OpenCV
image_path = 'tumor.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add noise to the original image using the provided noise functions
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(original_image, salt_prob=0.02, pepper_prob=0.02)
gaussian_noisy_image = add_gaussian_noise(original_image)
rayleigh_noisy_image = add_rayleigh_noise(original_image)

# Apply the weighted median filter to images with different types of noise
filtered_sp_image_wmf = weighted_median_filter(salt_and_pepper_noisy_image, weights)
filtered_gaussian_image_wmf = weighted_median_filter(gaussian_noisy_image, weights)
filtered_rayleigh_image_wmf = weighted_median_filter(rayleigh_noisy_image, weights)

# Apply the alpha trimmed mean filter to images with trim_threshold=1
filtered_sp_image_t1 = alpha_trimmed_mean_filter(salt_and_pepper_noisy_image, trim_threshold=1)
filtered_gaussian_image_t1 = alpha_trimmed_mean_filter(gaussian_noisy_image, trim_threshold=1)
filtered_rayleigh_image_t1 = alpha_trimmed_mean_filter(rayleigh_noisy_image, trim_threshold=1)

# Apply the alpha trimmed mean filter to images with trim_threshold=2
filtered_sp_image_t2 = alpha_trimmed_mean_filter(salt_and_pepper_noisy_image, trim_threshold=2)
filtered_gaussian_image_t2 = alpha_trimmed_mean_filter(gaussian_noisy_image, trim_threshold=2)
filtered_rayleigh_image_t2 = alpha_trimmed_mean_filter(rayleigh_noisy_image, trim_threshold=2)

# Calculate PSNR values for Alpha Trimmed Mean Filter with trim_threshold=1
psnr_sp_t1 = compute_psnr(original_image, filtered_sp_image_t1)
psnr_gaussian_t1 = compute_psnr(original_image, filtered_gaussian_image_t1)
psnr_rayleigh_t1 = compute_psnr(original_image, filtered_rayleigh_image_t1)

# Calculate PSNR values for Alpha Trimmed Mean Filter with trim_threshold=2
psnr_sp_t2 = compute_psnr(original_image, filtered_sp_image_t2)
psnr_gaussian_t2 = compute_psnr(original_image, filtered_gaussian_image_t2)
psnr_rayleigh_t2 = compute_psnr(original_image, filtered_rayleigh_image_t2)

# Calculate PSNR values for Weighted Median Filter
psnr_sp_wmf = compute_psnr(original_image, filtered_sp_image_wmf)
psnr_gaussian_wmf = compute_psnr(original_image, filtered_gaussian_image_wmf)
psnr_rayleigh_wmf = compute_psnr(original_image, filtered_rayleigh_image_wmf)

# Create a dictionary to organize PSNR values in a structured table format
psnr_data = {
    "Technique": [
        "Alpha Trimmed Mean Filter (t=1)",  # Describes the filter type and trimming threshold
        "Alpha Trimmed Mean Filter (t=2)",
        "Weighted Median Filter"
    ],
    "Salt and Pepper Noise": [psnr_sp_t1, psnr_sp_t2, psnr_sp_wmf],  # PSNR values for Salt and Pepper noise
    "Gaussian Noise": [psnr_gaussian_t1, psnr_gaussian_t2, psnr_gaussian_wmf],  # PSNR values for Gaussian noise
    "Rayleigh Noise": [psnr_rayleigh_t1, psnr_rayleigh_t2, psnr_rayleigh_wmf]  # PSNR values for Rayleigh noise
}

# Convert the dictionary to a DataFrame for displaying the PSNR table
df_psnr = pd.DataFrame(psnr_data)

# Print the PSNR values as a formatted table
print("PSNR Measurements Table:")
print(df_psnr.to_string(index=False))

# Plotting setup for visual comparison of the original noisy and filtered images
plt.figure(figsize=(18, 12))

# Plot Salt and Pepper noisy image and its filtered results
plt.subplot(3, 4, 1)
plt.title('Salt and Pepper Noisy Image')
plt.imshow(salt_and_pepper_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title('Filtered (Alpha Trimmed t=1)')
plt.imshow(filtered_sp_image_t1, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.title('Filtered (Alpha Trimmed t=2)')
plt.imshow(filtered_sp_image_t2, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.title('Filtered (Weighted Median)')
plt.imshow(filtered_sp_image_wmf, cmap='gray')
plt.axis('off')

# Plot Gaussian noisy image and its filtered results
plt.subplot(3, 4, 5)
plt.title('Gaussian Noisy Image')
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title('Filtered (Alpha Trimmed t=1)')
plt.imshow(filtered_gaussian_image_t1, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.title('Filtered (Alpha Trimmed t=2)')
plt.imshow(filtered_gaussian_image_t2, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.title('Filtered (Weighted Median)')
plt.imshow(filtered_gaussian_image_wmf, cmap='gray')
plt.axis('off')

# Plot Rayleigh noisy image and its filtered results
plt.subplot(3, 4, 9)
plt.title('Rayleigh Noisy Image')
plt.imshow(rayleigh_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title('Filtered (Alpha Trimmed t=1)')
plt.imshow(filtered_rayleigh_image_t1, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.title('Filtered (Alpha Trimmed t=2)')
plt.imshow(filtered_rayleigh_image_t2, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.title('Filtered (Weighted Median)')
plt.imshow(filtered_rayleigh_image_wmf, cmap='gray')
plt.axis('off')

# Adjust layout to ensure titles and images don't overlap
plt.tight_layout()
# Display the plots
plt.show()