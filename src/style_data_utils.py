"""Utilities for loading and preprocessing style images (artworks)."""

import urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image


# Famous public domain artworks (Wikimedia Commons)
STYLE_IMAGES = {
    'starry_night': {
        'name': 'The Starry Night - Van Gogh',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
        'artist': 'Vincent van Gogh'
    },
    'great_wave': {
        'name': 'The Great Wave - Hokusai',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/The_Great_Wave_off_Kanagawa.jpg/1280px-The_Great_Wave_off_Kanagawa.jpg',
        'artist': 'Katsushika Hokusai'
    },
    'scream': {
        'name': 'The Scream - Munch',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
        'artist': 'Edvard Munch'
    },
    'composition_vii': {
        'name': 'Composition VII - Kandinsky',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/1280px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
        'artist': 'Wassily Kandinsky'
    },
    'picasso_weeping': {
        'name': 'The Weeping Woman - Picasso',
        'url': 'https://upload.wikimedia.org/wikipedia/en/thumb/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg/800px-Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg',
        'artist': 'Pablo Picasso'
    },
    'mosaic': {
        'name': 'Byzantine Mosaic',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Meister_von_San_Vitale_in_Ravenna_004.jpg/800px-Meister_von_San_Vitale_in_Ravenna_004.jpg',
        'artist': 'Byzantine Art'
    },
    'klimt': {
        'name': 'The Kiss - Klimt',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Gustav_Klimt_016.jpg/800px-Gustav_Klimt_016.jpg',
        'artist': 'Gustav Klimt'
    },
    'monet': {
        'name': 'Water Lilies - Monet',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg',
        'artist': 'Claude Monet'
    },
    'ukiyo_e': {
        'name': 'Plum Estate - Hiroshige',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Hiroshige%2C_Plum_Park_in_Kameido.jpg/800px-Hiroshige%2C_Plum_Park_in_Kameido.jpg',
        'artist': 'Utagawa Hiroshige'
    },
    'abstract': {
        'name': 'Broadway Boogie Woogie - Mondrian',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Piet_Mondrian_-_Broadway_Boogie_Woogie_-_1943_-_MoMA.jpg/800px-Piet_Mondrian_-_Broadway_Boogie_Woogie_-_1943_-_MoMA.jpg',
        'artist': 'Piet Mondrian'
    }
}


def download_style_images(data_dir='../data/styles', target_size=(256, 256)):
    """
    Download famous artworks for style transfer.
    
    Args:
        data_dir: Directory to save style images
        target_size: Size to resize images to
    
    Returns:
        Dictionary mapping style keys to file paths
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    style_paths = {}
    
    print("Downloading style images...")
    for key, info in STYLE_IMAGES.items():
        file_path = data_path / f"{key}.jpg"
        
        if file_path.exists():
            print(f"  {info['name']} (already downloaded)")
        else:
            try:
                print(f"  ↓ Downloading {info['name']}...")
                urllib.request.urlretrieve(info['url'], file_path)
                
                # Resize and save
                img = Image.open(file_path)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(file_path, quality=95)
                
                print(f"    Saved to {file_path}")
            except Exception as e:
                print(f"    ✗ Failed to download {info['name']}: {e}")
                continue
        
        style_paths[key] = str(file_path)
    
    print(f"\nDownloaded {len(style_paths)} style images to {data_dir}")
    return style_paths


def load_style_images(data_dir='../data/styles', normalize=True):
    """
    Load all style images into memory.
    
    Args:
        data_dir: Directory containing style images
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Dictionary mapping style keys to numpy arrays
        Dictionary mapping style keys to metadata
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print("Style images not found. Downloading...")
        download_style_images(data_dir)
    
    style_images = {}
    style_metadata = {}
    
    for key, info in STYLE_IMAGES.items():
        file_path = data_path / f"{key}.jpg"
        
        if file_path.exists():
            # Load image
            img = Image.open(file_path).convert('RGB')
            img_array = np.array(img)
            
            if normalize:
                img_array = img_array.astype(np.float32) / 255.0
            
            style_images[key] = img_array
            style_metadata[key] = info
    
    print(f"Loaded {len(style_images)} style images")
    return style_images, style_metadata


def preprocess_style_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess a single style image.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image as numpy array (normalized to [0, 1])
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array
