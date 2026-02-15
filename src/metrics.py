"""
Metrics and evaluation utilities for autoencoders.
"""

# Third-party imports
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_auc_score, roc_curve


def calculate_mse(original, reconstructed):
    """
    Calculate Mean Squared Error between images.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
    
    Returns:
        MSE value(s)
    """
    return np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))


def calculate_ssim(original, reconstructed):
    """
    Calculate Structural Similarity Index (SSIM) between images.
    
    Args:
        original: Original images (batch or single)
        reconstructed: Reconstructed images
    
    Returns:
        SSIM value(s)
    """
    # Handle batch or single image
    if len(original.shape) == 4:
        # Batch of images
        ssim_values = []
        for i in range(len(original)):
            ssim_val = ssim(
                original[i],
                reconstructed[i],
                multichannel=True,
                channel_axis=2,
                data_range=1.0
            )
            ssim_values.append(ssim_val)
        return np.array(ssim_values)
    else:
        # Single image
        return ssim(
            original,
            reconstructed,
            multichannel=True,
            channel_axis=2,
            data_range=1.0
        )


def calculate_psnr(original, reconstructed, max_pixel=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        max_pixel: Maximum pixel value (default: 1.0 for normalized images)
    
    Returns:
        PSNR value(s) in dB
    """
    mse = calculate_mse(original, reconstructed)
    # Avoid division by zero
    mse = np.maximum(mse, 1e-10)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_reconstruction_error(model, images):
    """
    Calculate reconstruction error for anomaly detection.
    
    Args:
        model: Trained autoencoder model
        images: Input images
    
    Returns:
        Array of reconstruction errors (MSE per image)
    """
    reconstructed = model.predict(images, verbose=0)
    errors = calculate_mse(images, reconstructed)
    return errors


def compute_anomaly_threshold(errors, percentile=95):
    """
    Compute anomaly detection threshold based on percentile.
    
    Args:
        errors: Reconstruction errors from normal data
        percentile: Percentile for threshold (default: 95)
    
    Returns:
        Threshold value
    """
    return np.percentile(errors, percentile)


def compute_roc_metrics(y_true, scores):
    """
    Compute ROC curve and AUC score.
    
    Args:
        y_true: True binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
    
    Returns:
        Dictionary with fpr, tpr, thresholds, and auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc
    }


def find_optimal_threshold(y_true, scores):
    """
    Find optimal threshold using Youden's J statistic.
    
    Args:
        y_true: True binary labels
        scores: Anomaly scores
    
    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    
    return thresholds[optimal_idx]


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as 2D numpy array
    """
    return confusion_matrix(y_true, y_pred)


def calculate_compression_ratio(original_shape, latent_dim):
    """
    Calculate compression ratio.
    
    Args:
        original_shape: Shape of original image (H, W, C)
        latent_dim: Dimension of latent representation
    
    Returns:
        Compression ratio
    """
    original_size = np.prod(original_shape)
    compressed_size = latent_dim
    
    return original_size / compressed_size


def compute_metrics_summary(original, reconstructed, latent_dim):
    """
    Compute a summary of all quality metrics.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        latent_dim: Latent dimension for compression ratio
    
    Returns:
        Dictionary of metrics
    """
    mse = np.mean(calculate_mse(original, reconstructed))
    psnr = np.mean(calculate_psnr(original, reconstructed))
    ssim_value = np.mean(calculate_ssim(original, reconstructed))
    compression_ratio = calculate_compression_ratio(original.shape[1:], latent_dim)
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim_value),
        'compression_ratio': float(compression_ratio)
    }
