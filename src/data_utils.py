"""
Data utilities for loading and processing DF2K_OST image dataset.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


# HuggingFace dataset configuration (hardcoded)
DATASET_REPO_ID = "gperdrizet/DF2K_OST"


def load_df2k_ost(split='train', max_images=None):
    """
    Load DF2K_OST dataset from HuggingFace Hub.
    
    The dataset is automatically downloaded and cached on first use.
    All images are 256×256 RGB.
    
    Args:
        split: 'train' or 'validation'
        max_images: Maximum number of images to load (None = all)
    
    Returns:
        numpy array of shape (N, 256, 256, 3) with values in [0, 1]
    """
    print(f"Loading DF2K_OST dataset (split={split})")
    print(f"Repository: {DATASET_REPO_ID}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(DATASET_REPO_ID, split=split)
        
        if max_images is not None:
            dataset = dataset.select(range(min(max_images, len(dataset))))
        
        print(f"Processing {len(dataset)} images...")
        
        # Convert to numpy arrays
        images = []
        for sample in tqdm(dataset, desc="Loading images"):
            img = sample['image']  # PIL Image
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
        
        images = np.array(images)
        
        print(f"Loaded {len(images)} images")
        print(f"Shape: {images.shape}")
        print(f"Data type: {images.dtype}")
        print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
        
        return images
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print(f"\nMake sure the dataset exists at: https://huggingface.co/datasets/{DATASET_REPO_ID}")
        raise
    print(f"  Data type: {images.dtype}")
    print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    return images


def create_train_val_split(images, val_split=0.1, seed=42):
    """
    Split images into training and validation sets.
    
    Args:
        images: numpy array of images
        val_split: Fraction of data for validation (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        (train_images, val_images)
    """
    np.random.seed(seed)
    
    n_total = len(images)
    n_val = int(n_total * val_split)
    
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_images = images[train_indices]
    val_images = images[val_indices]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    
    return train_images, val_images


def add_gaussian_noise(images, noise_level=25):
    """
    Add Gaussian noise to images for denoising training.
    
    Args:
        images: numpy array of images in [0, 1]
        noise_level: Noise standard deviation (0-255 scale)
    
    Returns:
        Noisy images clipped to [0, 1]
    """
    noise_std = noise_level / 255.0
    noise = np.random.normal(0, noise_std, images.shape).astype(np.float32)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)
