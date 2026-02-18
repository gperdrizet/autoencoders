"""
Utilities for downloading pre-trained models from Hugging Face Hub.

This module provides functions to download trained autoencoder models from
Hugging Face, making it easy for students to use pre-trained models without
needing Git LFS or large file storage in the repository.
"""

from pathlib import Path
import os
from huggingface_hub import hf_hub_download

# Hugging Face repository configuration
# Update this with your actual Hugging Face username/organization
HF_REPO_ID = os.getenv('HF_REPO_ID', 'your-username/autoencoders-demo')

# Model files available in the repository
MODEL_FILES = {
    'compression_ae_latent32.keras': 'compression_ae_latent32.keras',
    'compression_ae_latent64.keras': 'compression_ae_latent64.keras',
    'compression_ae_latent128.keras': 'compression_ae_latent128.keras',
    'compression_ae_latent256.keras': 'compression_ae_latent256.keras',
    'anomaly_ae.keras': 'anomaly_ae.keras',
    'vae.keras': 'vae.keras',
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
