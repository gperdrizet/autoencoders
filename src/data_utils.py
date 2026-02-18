"""
Data utilities for loading and preprocessing TF Flowers dataset.
"""

# Standard library imports
import io

# Third-party imports
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


# Flower class names
FLOWER_CLASSES = [
    'dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'
]

# COCO class names (80 categories)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Image dimensions for models
IMAGE_SIZE = 64


def load_coco_cached(subset_percent=10, normalize=True, image_size=IMAGE_SIZE, cache_dir='../data'):
    """
    Load COCO dataset with caching for faster subsequent loads.
    
    This function attempts to load the dataset in the following order:
    1. Local cache (if exists)
    2. Hugging Face Hub (pre-processed subset)
    3. TensorFlow Datasets (full download)
    
    Args:
        subset_percent: Percentage of training data to use (1-100)
        normalize: If True, normalize pixel values to [0, 1]
        image_size: Target image size
        cache_dir: Directory to store cached subset
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test))
    """
    from pathlib import Path
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    cache_file = cache_path / f'coco_{subset_percent}percent_subset.npz'
    
    # Try loading from local cache first (silent - fast path)
    if cache_file.exists():
        data = np.load(cache_file)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        return (x_train, y_train), (x_test, y_test)
    
    # Try downloading from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        from src.huggingface_utils import HF_REPO_ID
        
        print(f'📥 Downloading COCO {subset_percent}% subset from HuggingFace...')
        
        # Download from HF Hub (has built-in progress bar)
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f'data/coco_{subset_percent}percent_subset.npz',
            repo_type='model',
            cache_dir=str(cache_path / '.hf_cache')
        )
        
        # Load and cache locally
        data = np.load(downloaded_path)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        
        np.savez_compressed(
            cache_file,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        
        return (x_train, y_train), (x_test, y_test)
        
    except Exception:
        # Fall back to TFDS
        print('📥 Downloading COCO dataset from TensorFlow Datasets (~95GB)...')
    
    # Fallback: Download from TFDS (has built-in progress bar)
    (x_train, y_train), (x_test, y_test) = load_coco(
        subset_percent=subset_percent,
        normalize=normalize,
        image_size=image_size
    )
    
    # Save to cache
    np.savez_compressed(
        cache_file,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    
    # Auto-upload to Hugging Face if token is available (instructor mode)
    try:
        from src.huggingface_utils import upload_dataset
        upload_dataset(cache_file, f'data/coco_{subset_percent}percent_subset.npz')
    except Exception:
        # Silent skip if upload fails - students don't need this
        pass
    
    return (x_train, y_train), (x_test, y_test)


def load_flowers(normalize=True, image_size=IMAGE_SIZE):
    """
    Load TF Flowers dataset.
    
    Downloads the dataset on first run and caches it locally.
    
    Args:
        normalize: If True, normalize pixel values to [0, 1]
        image_size: Target image size (images will be resized to image_size x image_size)
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test))
    """
    # Load dataset using TensorFlow Datasets
    # Split: 80% train, 20% test
    (ds_train, ds_test), ds_info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        as_supervised=True,
        with_info=True,
    )
    
    def preprocess_fn(image, label):
        """Resize and optionally normalize images."""
        image = tf.image.resize(image, [image_size, image_size])
        if normalize:
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.cast(image, tf.uint8)
        return image, label
    
    # Apply preprocessing
    ds_train = ds_train.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Convert to numpy arrays
    x_train = []
    y_train = []
    for image, label in ds_train:
        x_train.append(image.numpy())
        y_train.append(label.numpy())
    
    x_test = []
    y_test = []
    for image, label in ds_test:
        x_test.append(image.numpy())
        y_test.append(label.numpy())
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def load_coco(subset_percent=10, normalize=True, image_size=IMAGE_SIZE):
    """
    Load COCO 2017 dataset with a configurable subset.
    
    Downloads the dataset on first run and caches it locally.
    This is a large dataset (~25GB), so we use a subset by default.
    
    Args:
        subset_percent: Percentage of training data to use (1-100). Default is 10%.
                       Using 10% gives ~11,800 training images.
        normalize: If True, normalize pixel values to [0, 1]
        image_size: Target image size (images will be resized to image_size x image_size)
    
    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test))
        Note: y_train and y_test contain the label of the first object in each image
    """
    # Construct split strings
    train_split = f'train[:{subset_percent}%]'
    test_split = 'validation[:20%]'  # Use 20% of validation set for testing
    
    # Load dataset using TensorFlow Datasets
    # COCO is an object detection dataset, so we load it without as_supervised
    (ds_train, ds_test), ds_info = tfds.load(
        'coco/2017',
        split=[train_split, test_split],
        as_supervised=False,
        with_info=True,
    )
    
    def preprocess_fn(example):
        """Extract image and first object label, then resize and normalize."""
        image = example['image']
        
        # Extract the first object's label (COCO images can have multiple objects)
        # If no objects, use label 0
        objects = example['objects']
        label = tf.cond(
            tf.size(objects['label']) > 0,
            lambda: objects['label'][0],
            lambda: tf.constant(0, dtype=tf.int64)
        )
        
        # Resize image
        image = tf.image.resize(image, [image_size, image_size])
        
        # Normalize if requested
        if normalize:
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.cast(image, tf.uint8)
        
        return image, label
    
    # Apply preprocessing
    ds_train = ds_train.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Convert to numpy arrays
    x_train = []
    y_train = []
    for image, label in ds_train:
        x_train.append(image.numpy())
        y_train.append(label.numpy())
    
    x_test = []
    y_test = []
    for image, label in ds_test:
        x_test.append(image.numpy())
        y_test.append(label.numpy())
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def create_anomaly_dataset(normal_classes=[0, 1, 2, 3], normalize=True):
    """
    Create anomaly detection dataset.
    
    Splits flowers dataset into 'normal' and 'anomalous' classes.
    
    Args:
        normal_classes: List of class indices considered normal (default: first 4 classes)
        normalize: If True, normalize pixel values to [0, 1]
    
    Returns:
        Dictionary with train/test splits for normal and anomaly data
    """
    (x_train, y_train), (x_test, y_test) = load_flowers(normalize=normalize)
    
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


def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
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
