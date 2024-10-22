import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    salt_num = np.ceil(salt_prob * total_pixels).astype(int)
    pepper_num = np.ceil(pepper_prob * total_pixels).astype(int)
    
    # add salt (white pixels)
    coords = [np.random.randint(0, i - 1, salt_num) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # add pepper (black pixels)
    coords = [np.random.randint(0, i - 1, pepper_num) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def add_rayleigh_noise(image, scale=25):
    noise = np.random.rayleigh(scale, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def alpha_trimmed_mean_filter(image, kernel_size=5, t=1):
    padded_image = np.pad(image, (kernel_size // 2, kernel_size // 2), mode='edge')
    output_image = np.zeros(image.shape, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # extract the kernel
            kernel = padded_image[i:i + kernel_size, j:j + kernel_size].flatten()
            # sort the values
            sorted_kernel = np.sort(kernel)
            # trim the t minimum and t maximum values
            trimmed_kernel = sorted_kernel[t:-(t + 1)]
            # find the mean of the kernal trimmed
            output_image[i, j] = np.mean(trimmed_kernel)

    return np.clip(output_image, 0, 255).astype(np.uint8)


image_path = 'tumor.png' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)
gaussian_noisy_image = add_gaussian_noise(image)
rayleigh_noisy_image = add_rayleigh_noise(image)

filtered_sp_image_t1 = alpha_trimmed_mean_filter(salt_and_pepper_noisy_image, t=1)
filtered_sp_image_t2 = alpha_trimmed_mean_filter(salt_and_pepper_noisy_image, t=2)

filtered_gaussian_image_t1 = alpha_trimmed_mean_filter(gaussian_noisy_image, t=1)
filtered_gaussian_image_t2 = alpha_trimmed_mean_filter(gaussian_noisy_image, t=2)

filtered_rayleigh_image_t1 = alpha_trimmed_mean_filter(rayleigh_noisy_image, t=1)
filtered_rayleigh_image_t2 = alpha_trimmed_mean_filter(rayleigh_noisy_image, t=2)

plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# salt and pepper Noisy Image
plt.subplot(3, 3, 2)
plt.title('Salt and Pepper Noisy Image')
plt.imshow(salt_and_pepper_noisy_image, cmap='gray')
plt.axis('off')

# filtered images for salt and pepper
plt.subplot(3, 3, 3)
plt.title('Filtered (t=1)')
plt.imshow(filtered_sp_image_t1, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title('Filtered (t=2)')
plt.imshow(filtered_sp_image_t2, cmap='gray')
plt.axis('off')

# gaussian noisy image
plt.subplot(3, 3, 5)
plt.title('Gaussian Noisy Image')
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.axis('off')

# filitered images for Gaussian
plt.subplot(3, 3, 6)
plt.title('Filtered (t=1)')
plt.imshow(filtered_gaussian_image_t1, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.title('Filtered (t=2)')
plt.imshow(filtered_gaussian_image_t2, cmap='gray')
plt.axis('off')

# rayleigh noisy Image
plt.subplot(3, 3, 8)
plt.title('Rayleigh Noisy Image')
plt.imshow(rayleigh_noisy_image, cmap='gray')
plt.axis('off')

# filtered Images for Rayleigh
plt.subplot(3, 3, 9)
plt.title('Filtered (t=1)')
plt.imshow(filtered_rayleigh_image_t1, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
