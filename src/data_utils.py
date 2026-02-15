"""
Data utilities for loading and preprocessing CIFAR-10 dataset.
"""

# Standard library imports
import io

# Third-party imports
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import cifar10


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_cifar10(normalize=True):
    """
    Load CIFAR-10 dataset.
    
    Downloads the dataset on first run and caches it locally.
    
    Args:
        normalize: If True, normalize pixel values to [0, 1]
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test))
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (x_train, y_train), (x_test, y_test)


def create_anomaly_dataset(normal_classes=[0, 1, 2, 3, 4, 5, 6, 7], normalize=True):
    """
    Create anomaly detection dataset.
    
    Splits CIFAR-10 into 'normal' and 'anomalous' classes.
    
    Args:
        normal_classes: List of class indices considered normal
        normalize: If True, normalize pixel values to [0, 1]
    
    Returns:
        Dictionary with train/test splits for normal and anomaly data
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10(normalize=normalize)
    
    # Create masks for normal classes
    train_normal_mask = np.isin(y_train, normal_classes)
    test_normal_mask = np.isin(y_test, normal_classes)
    
    # Create masks for anomalous classes
    train_anomaly_mask = ~train_normal_mask
    test_anomaly_mask = ~test_normal_mask
    
    return {
        'x_train_normal': x_train[train_normal_mask],
        'y_train_normal': y_train[train_normal_mask],
        'x_train_anomaly': x_train[train_anomaly_mask],
        'y_train_anomaly': y_train[train_anomaly_mask],
        'x_test_normal': x_test[test_normal_mask],
        'y_test_normal': y_test[test_normal_mask],
        'x_test_anomaly': x_test[test_anomaly_mask],
        'y_test_anomaly': y_test[test_anomaly_mask],
    }


def preprocess_image(image, target_size=(32, 32)):
    """
    Preprocess a single image for model input.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed numpy array with shape (1, H, W, 3) and values in [0, 1]
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Resize if needed
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def upload_and_preprocess(uploaded_file):
    """
    Process an uploaded file for Streamlit.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Preprocessed numpy array
    """
    # Read image from uploaded file
    image = Image.open(uploaded_file)
    
    # Preprocess
    return preprocess_image(image)


def get_random_samples(x_data, y_data=None, n_samples=10, seed=None):
    """
    Get random samples from dataset.
    
    Args:
        x_data: Image data
        y_data: Labels (optional)
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (images, labels) or just images if y_data is None
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.choice(len(x_data), size=n_samples, replace=False)
    
    if y_data is not None:
        return x_data[indices], y_data[indices]
    else:
        return x_data[indices]


def get_samples_by_class(x_data, y_data, class_id, n_samples=10):
    """
    Get samples of a specific class.
    
    Args:
        x_data: Image data
        y_data: Labels
        class_id: Class index
        n_samples: Number of samples to return
    
    Returns:
        Numpy array of images
    """
    class_mask = y_data == class_id
    class_images = x_data[class_mask]
    
    if len(class_images) < n_samples:
        n_samples = len(class_images)
    
    indices = np.random.choice(len(class_images), size=n_samples, replace=False)
    
    return class_images[indices]


def denormalize_image(image):
    """
    Convert normalized [0, 1] image back to [0, 255] uint8.
    
    Args:
        image: Normalized numpy array
    
    Returns:
        Denormalized uint8 array
    """
    return (image * 255).astype(np.uint8)


def numpy_to_pil(image):
    """
    Convert numpy array to PIL Image.
    
    Args:
        image: Numpy array (can be normalized or uint8)
    
    Returns:
        PIL Image
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = denormalize_image(image)
    
    return Image.fromarray(image)
