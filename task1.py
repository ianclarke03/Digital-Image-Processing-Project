import numpy as np
import cv2
import matplotlib.pyplot as plt


#function for generating Rayleigh noise
def generate_rayleigh_noise(image, scale_param):

    #uniformly distributed random variable between 0 and 1
    uniform_random = np.random.uniform(0, 1, image.shape)
    
    #generate Rayleigh noise using the formula n = b * sqrt(-2 * ln(1 - u))
    rayleigh_noise = scale_param * np.sqrt(-2 * np.log(1 - uniform_random))
    
    return rayleigh_noise




#function for adding Rayleigh noise to an image
def add_rayleigh_noise(image, scale_param):

    #normalize the image to range [0, 1] if it's not already (to make it easier to add the noise to the image)
    if image.max() > 1:
        image_normalized = image / 255.0
    else: 
        image
    
    #generate the Rayleigh noise and add it to the image
    noise = generate_rayleigh_noise(image_normalized, scale_param)
    noisy_image = image_normalized + noise
    
    #clip the values to stay within [0, 1] and convert back to range [0,255]
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    

    return noisy_image


image = cv2.imread('tumor.png', cv2.IMREAD_GRAYSCALE)

noisy_image = add_rayleigh_noise(image, 0.5) #adding Rayleigh noise to the image with a scale parameter of 0.5, higher scale param = more noise


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Noisy Image (Rayleigh Noise)')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.show()
