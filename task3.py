# task3.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from task2 import add_salt_and_pepper_noise, add_gaussian_noise, add_rayleigh_noise, alpha_trimmed_mean_filter

def weighted_median_filter(image, weights):
    # Pad the image to handle borders
    padded_image = np.pad(image, (1, 1), mode='edge')
    output_image = np.zeros_like(image)

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the 3x3 kernel
            kernel = padded_image[i:i + 3, j:j + 3]
            # Flatten the kernel and repeat values according to weights
            weighted_values = np.repeat(kernel.flatten(), weights.flatten())
            # Sort the weighted values
            sorted_values = np.sort(weighted_values)
            # Find the median
            median_value = np.median(sorted_values)
            # Assign the median value to the output image
            output_image[i, j] = median_value

    return output_image.astype(np.uint8)


# Weights for the 3x3 kernel
weights = np.array([
    [1, 2, 1],
    [2, 3, 2],
    [1, 2, 1]
])

# Load the original grayscale image
image_path = 'tumor.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add noise to the image
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)
gaussian_noisy_image = add_gaussian_noise(image)
rayleigh_noisy_image = add_rayleigh_noise(image)

# Apply the weighted median filter to noisy images
filtered_sp_image_wmf = weighted_median_filter(salt_and_pepper_noisy_image, weights)
filtered_gaussian_image_wmf = weighted_median_filter(gaussian_noisy_image, weights)
filtered_rayleigh_image_wmf = weighted_median_filter(rayleigh_noisy_image, weights)

# Compare with Alpha Trimmed Mean Filter
filtered_sp_image_t1 = alpha_trimmed_mean_filter(salt_and_pepper_noisy_image, trim_threshold=1)
filtered_gaussian_image_t1 = alpha_trimmed_mean_filter(gaussian_noisy_image, trim_threshold=1)
filtered_rayleigh_image_t1 = alpha_trimmed_mean_filter(rayleigh_noisy_image, trim_threshold=1)

# Plot the original and filtered images
plt.figure(figsize=(18, 12))

# Original Noisy and Filtered with Weighted Median Filter
plt.subplot(3, 4, 1)
plt.title('Salt and Pepper Noisy Image')
plt.imshow(salt_and_pepper_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title('Weighted Median Filtered')
plt.imshow(filtered_sp_image_wmf, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.title('Alpha Trimmed Filter (threshold=1)')
plt.imshow(filtered_sp_image_t1, cmap='gray')
plt.axis('off')

# Gaussian Noisy and Filtered with Weighted Median Filter
plt.subplot(3, 4, 5)
plt.title('Gaussian Noisy Image')
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title('Weighted Median Filtered')
plt.imshow(filtered_gaussian_image_wmf, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.title('Alpha Trimmed Filter (threshold=1)')
plt.imshow(filtered_gaussian_image_t1, cmap='gray')
plt.axis('off')

# Rayleigh Noisy and Filtered with Weighted Median Filter
plt.subplot(3, 4, 9)
plt.title('Rayleigh Noisy Image')
plt.imshow(rayleigh_noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title('Weighted Median Filtered')
plt.imshow(filtered_rayleigh_image_wmf, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.title('Alpha Trimmed Filter (threshold=1)')
plt.imshow(filtered_rayleigh_image_t1, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
