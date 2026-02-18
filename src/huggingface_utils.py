"""
Utilities for downloading and uploading models/datasets to Hugging Face Hub.

This module provides functions to:
- Download pre-trained autoencoder models from Hugging Face
- Upload trained models to Hugging Face (requires HF_TOKEN)
- Upload/download datasets (COCO subset)

Students can use pre-trained models without configuration. Instructors can
upload their own models by setting HF_REPO_ID and HF_TOKEN in .env file.
"""

from pathlib import Path
import os
from huggingface_hub import hf_hub_download, HfApi

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Hugging Face repository configuration
# Default: Uses public pre-trained models from mrdbourke/autoencoders-demo
# To upload your own models: set HF_REPO_ID and HF_TOKEN in .env file
HF_REPO_ID = os.getenv('HF_REPO_ID', 'mrdbourke/autoencoders-demo')
HF_TOKEN = os.getenv('HF_TOKEN', None)

# Model files available in the repository
MODEL_FILES = {
    'compression_ae_latent32.keras': 'compression_ae_latent32.keras',
    'compression_ae_latent64.keras': 'compression_ae_latent64.keras',
    'compression_ae_latent128.keras': 'compression_ae_latent128.keras',
    'compression_ae_latent256.keras': 'compression_ae_latent256.keras',
    'anomaly_ae.keras': 'anomaly_ae.keras',
    'denoising_ae.keras': 'denoising_ae.keras',
}


def download_model(model_name, models_dir='models', use_cache=True):
    """
    Download a model from Hugging Face Hub if it doesn't exist locally.
    
    Args:
        model_name (str): Name of the model file to download
        models_dir (str): Local directory to save the model (default: 'models')
        use_cache (bool): Whether to use Hugging Face cache (default: True)
    
    Returns:
        str: Path to the downloaded model file
        
    Raises:
        RuntimeError: If the model download fails
        ValueError: If the model name is not in the available models list
    """
    if model_name not in MODEL_FILES:
        available = ', '.join(MODEL_FILES.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    local_path = models_path / model_name
    
    # Return if already downloaded
    if local_path.exists():
        return str(local_path)
    
    # Download from Hugging Face
    try:
        print(f'📥 Downloading {model_name} from Hugging Face...')
        
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILES[model_name],
            cache_dir=str(models_path / '.cache') if use_cache else None,
            local_dir=str(models_path),
            local_dir_use_symlinks=False
        )
        
        print(f'Downloaded successfully: {downloaded_path}')
        return downloaded_path
        
    except Exception as e:
        raise RuntimeError(f'Failed to download {model_name} from Hugging Face: {e}')


def ensure_model_available(model_name, models_dir='models'):
    """
    Ensure a model is available locally, downloading if necessary.
    
    This is a convenience wrapper around download_model that provides
    a simpler interface for checking and downloading models.
    
    Args:
        model_name (str): Name of the model file
        models_dir (str): Local directory to save the model (default: 'models')
    
    Returns:
        str: Path to the model file
    """
    models_path = Path(models_dir)
    local_path = models_path / model_name
    
    if local_path.exists():
        return str(local_path)
    
    return download_model(model_name, models_dir)


def list_available_models():
    """
    List all models available for download from Hugging Face.
    
    Returns:
        list: List of available model filenames
    """
    return list(MODEL_FILES.keys())


def check_local_models(models_dir='models'):
    """
    Check which models are available locally.
    
    Args:
        models_dir (str): Local directory containing models (default: 'models')
    
    Returns:
        dict: Dictionary with model names as keys and boolean values indicating availability
    """
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    local_models = {}
    for model_name in MODEL_FILES.keys():
        local_path = models_path / model_name
        local_models[model_name] = local_path.exists()
    
    return local_models


def upload_model(model_path, repo_path=None):
    """
    Upload a trained model to Hugging Face Hub.
    
    Requires HF_TOKEN to be set in environment (students don't need this).
    Silently skips upload if token is not available.
    
    Args:
        model_path (str or Path): Path to the local model file
        repo_path (str, optional): Path in HF repo. If None, uses model filename
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    if not HF_TOKEN:
        # Silent skip - students don't need to upload
        return False
    
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f'⚠️  Model file not found: {model_path}')
            return False
        
        if repo_path is None:
            repo_path = model_path.name
        
        print(f'📤 Uploading {model_path.name} to Hugging Face...')
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            repo_type='model'
        )
        
        print(f'✓ Successfully uploaded to {HF_REPO_ID}/{repo_path}')
        return True
        
    except Exception as e:
        print(f'⚠️  Upload failed: {e}')
        return False


def upload_dataset(file_path, repo_path):
    """
    Upload a dataset file to Hugging Face Hub.
    
    Requires HF_TOKEN to be set in environment (students don't need this).
    Silently skips upload if token is not available.
    
    Args:
        file_path (str or Path): Path to the local dataset file
        repo_path (str): Path in HF repo (e.g., 'data/coco_10percent_subset.npz')
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    if not HF_TOKEN:
        # Silent skip - students don't need to upload
        return False
    
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f'⚠️  Dataset file not found: {file_path}')
            return False
        
        print(f'📤 Uploading {file_path.name} to Hugging Face...')
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            repo_type='model'
        )
        
        print(f'✓ Successfully uploaded to {HF_REPO_ID}/{repo_path}')
        return True
        
    except Exception as e:
        print(f'⚠️  Upload failed: {e}')
        return False
