"""
Noise generation functions for image augmentation and denoising experiments.
"""

import numpy as np


def add_gaussian_noise(images, noise_factor=0.3):
    """
    Add Gaussian (normal) noise to images.
    
    Args:
        images: Clean images (numpy array)
        noise_factor: Standard deviation of noise (higher = more noise)
    
    Returns:
        Noisy images clipped to [0, 1]
    """
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0)


def add_salt_pepper_noise(images, amount=0.05):
    """
    Add salt and pepper noise (random black and white pixels).
    
    Args:
        images: Clean images (numpy array)
        amount: Fraction of pixels to corrupt
    
    Returns:
        Noisy images
    """
    noisy_images = images.copy()
    
    # Salt (white pixels)
    num_salt = int(amount * images.size * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in images.shape]
    noisy_images[coords[0], coords[1], coords[2], coords[3]] = 1.0
    
    # Pepper (black pixels)
    num_pepper = int(amount * images.size * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in images.shape]
    noisy_images[coords[0], coords[1], coords[2], coords[3]] = 0.0
    
    return noisy_images


def add_speckle_noise(images, noise_factor=0.2):
    """
    Add speckle (multiplicative) noise.
    
    Args:
        images: Clean images (numpy array)
        noise_factor: Scale of noise
    
    Returns:
        Noisy images clipped to [0, 1]
    """
    noise = np.random.randn(*images.shape) * noise_factor
    noisy_images = images + images * noise
    return np.clip(noisy_images, 0.0, 1.0)
