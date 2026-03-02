"""
Data utilities for loading and processing DF2K_OST image dataset.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import tensorflow as tf


# HuggingFace dataset configuration (hardcoded)
dataset_repo_id = "gperdrizet/DF2K_OST"


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
    print(f"Repository: {dataset_repo_id}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_repo_id, split=split)
        
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
        print(f"\nError loading dataset: {e}")
        print(f"\nMake sure the dataset exists at: https://huggingface.co/datasets/{dataset_repo_id}")
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


def create_tf_dataset(split='train', batch_size=16, shuffle=True, val_split=0.1, seed=42):
    """
    Create TensorFlow dataset that loads images on-the-fly (memory efficient).
    
    This function loads images in batches directly from HuggingFace without
    loading the entire dataset into memory first.
    
    Args:
        split: 'train' or 'validation'
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        val_split: Fraction for validation split (only used if split='train')
        seed: Random seed for reproducibility
    
    Returns:
        (train_dataset, val_dataset, dataset_info) where dataset_info contains sizes
    """
    print(f"Creating TensorFlow dataset from {dataset_repo_id} (split={split})")
    
    # Load HuggingFace dataset (lazy-loaded, no memory overhead)
    hf_dataset = load_dataset(dataset_repo_id, split=split)
    total_size = len(hf_dataset)
    
    print(f"Total images: {total_size}")
    
    # Split into train/val
    if val_split > 0:
        split_idx = int(total_size * (1 - val_split))
        
        # Shuffle indices for split
        indices = np.random.RandomState(seed).permutation(total_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_hf = hf_dataset.select(train_indices)
        val_hf = hf_dataset.select(val_indices)
        
        print(f"Split: {len(train_hf)} train, {len(val_hf)} validation")
    else:
        train_hf = hf_dataset
        val_hf = None
        print(f"No split: {len(train_hf)} images")
    
    def image_generator(hf_dataset):
        """Generator that yields (image, image) pairs for autoencoder training."""
        for sample in hf_dataset:
            img = sample['image']  # PIL Image
            img_array = np.array(img, dtype=np.float32) / 255.0
            yield img_array, img_array
    
    # Create train dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(train_hf),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        )
    )
    
    train_dataset = train_dataset.repeat()  # Repeat indefinitely for multiple epochs
    
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=256, seed=seed)
    
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    if val_hf is not None:
        val_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator(val_hf),
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
            )
        )
        val_dataset = val_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None
    
    dataset_info = {
        'train_size': len(train_hf),
        'val_size': len(val_hf) if val_hf else 0,
        'total_size': total_size
    }
    
    return train_dataset, val_dataset, dataset_info


def create_denoising_tf_dataset(split='train', batch_size=16, noise_level=25, 
                                shuffle=True, val_split=0.1, seed=42):
    """
    Create TensorFlow dataset for denoising (loads images on-the-fly with noise).
    
    Args:
        split: 'train' or 'validation'
        batch_size: Batch size for training
        noise_level: Gaussian noise sigma (0-255 scale)
        shuffle: Whether to shuffle the dataset
        val_split: Fraction for validation split
        seed: Random seed for reproducibility
    
    Returns:
        (train_dataset, val_dataset, dataset_info)
    """
    print(f"Creating denoising dataset from {dataset_repo_id} (split={split})")
    
    hf_dataset = load_dataset(dataset_repo_id, split=split)
    total_size = len(hf_dataset)
    
    print(f"Total images: {total_size}")
    print(f"Noise level: σ={noise_level}")
    
    # Split into train/val
    if val_split > 0:
        split_idx = int(total_size * (1 - val_split))
        indices = np.random.RandomState(seed).permutation(total_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_hf = hf_dataset.select(train_indices)
        val_hf = hf_dataset.select(val_indices)
        
        print(f"Split: {len(train_hf)} train, {len(val_hf)} validation")
    else:
        train_hf = hf_dataset
        val_hf = None
    
    noise_std = noise_level / 255.0
    
    def denoising_generator(hf_dataset, noise_std):
        """Generator that yields (noisy_image, clean_image) pairs."""
        for sample in hf_dataset:
            img = sample['image']
            clean = np.array(img, dtype=np.float32) / 255.0
            
            # Add noise
            noise = np.random.normal(0, noise_std, clean.shape).astype(np.float32)
            noisy = np.clip(clean + noise, 0, 1)
            
            yield noisy, clean
    
    # Create train dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: denoising_generator(train_hf, noise_std),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
        )
    )
    
    train_dataset = train_dataset.repeat()  # Repeat indefinitely for multiple epochs
    
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=256, seed=seed)
    
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    if val_hf is not None:
        val_dataset = tf.data.Dataset.from_generator(
            lambda: denoising_generator(val_hf, noise_std),
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
            )
        )
        val_dataset = val_dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None
    
    dataset_info = {
        'train_size': len(train_hf),
        'val_size': len(val_hf) if val_hf else 0,
        'total_size': total_size
    }
    
    return train_dataset, val_dataset, dataset_info

