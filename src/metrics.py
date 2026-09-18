"""
Metrics for evaluating autoencoder performance.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_mse(original, reconstructed):
    """
    Calculate Mean Squared Error.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
    
    Returns:
        MSE value
    """
    return np.mean((original - reconstructed) ** 2)


def calculate_psnr(original, reconstructed, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        original: Original images  
        reconstructed: Reconstructed images
        data_range: Range of data (1.0 for normalized images)
    
    Returns:
        PSNR in dB
    """
    # Calculate for each image and return mean
    psnr_values = []
    
    for i in range(len(original)):
        psnr_val = psnr(original[i], reconstructed[i], data_range=data_range)
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def calculate_ssim(original, reconstructed, data_range=1.0):
    """
    Calculate Structural Similarity Index.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images  
        data_range: Range of data (1.0 for normalized images)
    
    Returns:
        Mean SSIM value
    """
    ssim_values = []
    
    for i in range(len(original)):
        ssim_val = ssim(
            original[i], 
            reconstructed[i], 
            data_range=data_range,
            channel_axis=2  # Color channel is last dimension
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def compute_metrics(original, reconstructed, latent_dim=None):
    """
    Compute all metrics for reconstructed images.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        latent_dim: Latent dimension (for compression ratio calculation)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': calculate_mse(original, reconstructed),
        'psnr': calculate_psnr(original, reconstructed),
        'ssim': calculate_ssim(original, reconstructed),
    }
    
    if latent_dim is not None:
        input_size = original.shape[1] * original.shape[2] * original.shape[3]
        metrics['compression_ratio'] = input_size / latent_dim
    
    return metrics
