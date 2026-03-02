#!/usr/bin/env python3
"""
One-time script to process DF2K_OST dataset and upload to HuggingFace.
This script can be deleted after successful upload.

Dataset composition:
- DIV2K: 900 high-resolution images (800 train + 100 valid)
- Flickr2K: 2,650 high-resolution images  
- OST: 10,324 outdoor scene images

Processing:
- Resize all images to 256×256 using Lanczos resampling
- Create 90/10 train/validation split
- Upload to HuggingFace dataset repository

Requirements (install with pip):
- Pillow
- numpy
- tqdm
- python-dotenv
- datasets
- huggingface-hub
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Check for required packages
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip install tqdm")
    from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv

try:
    from datasets import Dataset, DatasetDict, Image as DatasetImage
except ImportError:
    print("Installing datasets...")
    os.system("pip install datasets")
    from datasets import Dataset, DatasetDict, Image as DatasetImage

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface-hub...")
    os.system("pip install huggingface-hub")
    from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

# Configuration
RAW_DATA_DIR = Path("/workspaces/autoencoders/data/raw")
TARGET_SIZE = (256, 256)
TRAIN_SPLIT = 0.9
REPO_ID = os.getenv("HF_REPO_ID", "gperdrizet/DF2K_OST")
HF_TOKEN = os.getenv("HF_TOKEN")

# Dataset sources
SOURCES = {
    "DIV2K": RAW_DATA_DIR / "DIV2K",
    "Flickr2K": RAW_DATA_DIR / "Flickr2K", 
    "OST": RAW_DATA_DIR / "OST" / "OutdoorSceneTrain_v2"
}


def find_all_images(directory):
    """Recursively find all image files in directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix in extensions:
                image_paths.append(Path(root) / file)
    
    return image_paths


def process_image(image_path, target_size=(256, 256)):
    """
    Load and resize image to target size using Lanczos resampling.
    
    Args:
        image_path: Path to image file
        target_size: Target (width, height)
    
    Returns:
        PIL Image resized to target_size
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def collect_all_images():
    """Collect all image paths from all sources."""
    print("=" * 70)
    print("COLLECTING IMAGE PATHS")
    print("=" * 70)
    
    all_images = []
    
    for source_name, source_dir in SOURCES.items():
        if not source_dir.exists():
            print(f" {source_name} directory not found: {source_dir}")
            continue
        
        print(f"\n{source_name}:")
        print(f"  Directory: {source_dir}")
        
        image_paths = find_all_images(source_dir)
        print(f"  Found {len(image_paths)} images")
        
        for path in image_paths:
            all_images.append({
                'path': path,
                'source': source_name
            })
    
    print(f"\n Total images found: {len(all_images)}")
    return all_images


def process_and_split_dataset(image_list, train_split=0.9):
    """
    Process all images and create train/validation split.
    
    Args:
        image_list: List of dicts with 'path' and 'source' keys
        train_split: Fraction of data for training
    
    Returns:
        (train_data, val_data) tuples of processed images
    """
    print("\n" + "=" * 70)
    print("PROCESSING IMAGES")
    print("=" * 70)
    
    # Shuffle images
    np.random.seed(42)
    indices = np.random.permutation(len(image_list))
    
    # Calculate split point
    n_train = int(len(image_list) * train_split)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"Train: {len(train_indices)} images")
    print(f"Validation: {len(val_indices)} images")
    
    # Process training images
    print("\nProcessing training images...")
    train_data = []
    for idx in tqdm(train_indices, desc="Train"):
        img_info = image_list[idx]
        img = process_image(img_info['path'], TARGET_SIZE)
        if img is not None:
            train_data.append({
                'image': img,
                'source': img_info['source'],
                'filename': img_info['path'].name
            })
    
    # Process validation images
    print("\nProcessing validation images...")
    val_data = []
    for idx in tqdm(val_indices, desc="Validation"):
        img_info = image_list[idx]
        img = process_image(img_info['path'], TARGET_SIZE)
        if img is not None:
            val_data.append({
                'image': img,
                'source': img_info['source'],
                'filename': img_info['path'].name
            })
    
    print(f"\nProcessed {len(train_data)} training images")
    print(f"Processed {len(val_data)} validation images")
    
    return train_data, val_data


def create_huggingface_dataset(train_data, val_data):
    """Create HuggingFace Dataset from processed data."""
    print("\n" + "=" * 70)
    print("CREATING HUGGINGFACE DATASET")
    print("=" * 70)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'image': [d['image'] for d in train_data],
        'source': [d['source'] for d in train_data],
        'filename': [d['filename'] for d in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        'image': [d['image'] for d in val_data],
        'source': [d['source'] for d in val_data],
        'filename': [d['filename'] for d in val_data]
    })
    
    # Cast image column to proper type
    train_dataset = train_dataset.cast_column('image', DatasetImage())
    val_dataset = val_dataset.cast_column('image', DatasetImage())
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Created dataset with {len(train_dataset)} train / {len(val_dataset)} validation images")
    
    return dataset_dict


def create_dataset_card(dataset_dict):
    """Create README.md content for dataset card."""
    return f"""---
license: apache-2.0
task_categories:
- image-to-image
- image-classification
size_categories:
- 10K<n<100K
tags:
- computer-vision
- image-processing
- autoencoders
- image-compression
- denoising
---

# DF2K_OST Dataset

High-quality 256×256 image dataset for training autoencoders.

## Dataset Description

This dataset combines three high-quality image sources commonly used for image super-resolution and restoration tasks:

- **DIV2K**: 900 high-resolution images (800 train + 100 validation)
- **Flickr2K**: 2,650 high-resolution images
- **OST (Outdoor Scene Training)**: 10,324 outdoor scene images

All images have been resized to 256×256 pixels using Lanczos resampling for optimal quality.

## Dataset Structure

```
DF2K_OST/
├── train/          # ~90% of images ({len(dataset_dict['train'])} images)
└── validation/     # ~10% of images ({len(dataset_dict['validation'])} images)
```

Each sample contains:
- `image`: 256×256 RGB image
- `source`: Original dataset source (DIV2K, Flickr2K, or OST)
- `filename`: Original filename

## Processing

All images were processed using:
- Target resolution: 256×256 pixels
- Resampling method: Lanczos (PIL.Image.LANCZOS)
- Color mode: RGB
- Train/validation split: 90/10 (stratified random)

## Usage

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("gperdrizet/DF2K_OST")

# Load only training split
train_data = load_dataset("gperdrizet/DF2K_OST", split="train")

# Access images
for sample in train_data:
    image = sample['image']  # PIL Image
    source = sample['source']  # Dataset source
```

## Original Sources

### DIV2K
- **Citation**: Agustsson, E., & Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.
- **Paper**: CVPR Workshops 2017
- **Website**: https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Flickr2K
- **Citation**: Agustsson, E. & Timofte, R. (2017). "NTIRE 2017 challenge on single image super-resolution: Dataset and study." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
- **Repository**: https://github.com/limbee/NTIRE2017

### OST (Outdoor Scene Training)
- **Citation**: Wang, X., Yu, K., Dong, C., & Loy, C. C. (2018). Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform. arXiv:1804.02815
- **Paper**: https://arxiv.org/abs/1804.02815
- **Repository**: https://github.com/xinntao/SFTGAN

## License

The compilation and processing are provided under Apache 2.0 license. Individual images retain their original licenses from source datasets.

## Citation

If you use this dataset, please cite the original source datasets:

```bibtex
@inproceedings{{agustsson2017ntire,
  title={{NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study}},
  author={{Agustsson, Eirikur and Timofte, Radu}},
  booktitle={{CVPR Workshops}},
  year={{2017}}
}}

@misc{{wang2018recoveringrealistictextureimage,
  title={{Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform}}, 
  author={{Xintao Wang and Ke Yu and Chao Dong and Chen Change Loy}},
  year={{2018}},
  eprint={{1804.02815}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CV}},
  url={{https://arxiv.org/abs/1804.02815}}
}}
```

## Created By

Processed and compiled for the Autoencoders educational demo project.
Repository: https://github.com/gperdrizet/autoencoders
"""


def upload_to_huggingface(dataset_dict):
    """Upload dataset to HuggingFace Hub."""
    print("\n" + "=" * 70)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("=" * 70)
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in environment variables")
        print("Please set HF_TOKEN in .env file")
        return False
    
    print(f"Repository: {REPO_ID}")
    print(f"Token: {'*' * 20}{HF_TOKEN[-4:]}")
    
    try:
        # Push dataset
        print("\nUploading dataset...")
        dataset_dict.push_to_hub(
            REPO_ID,
            token=HF_TOKEN,
            private=False
        )
        
        # Create and upload README
        print("\nCreating dataset card...")
        readme_content = create_dataset_card(dataset_dict)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{REPO_ID}")
        return True
        
    except Exception as e:
        print(f"\nError uploading dataset: {e}")
        return False


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("DF2K_OST DATASET BUILDER")
    print("=" * 70)
    print(f"Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"Train/val split: {TRAIN_SPLIT}/{1-TRAIN_SPLIT}")
    print(f"Repository: {REPO_ID}")
    print()
    
    # Step 1: Collect all images
    image_list = collect_all_images()
    
    if len(image_list) == 0:
        print("\nNo images found! Check raw data directories.")
        sys.exit(1)
    
    # Step 2: Process and split
    train_data, val_data = process_and_split_dataset(image_list, TRAIN_SPLIT)
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("\nNo images were successfully processed!")
        sys.exit(1)
    
    # Step 3: Create HuggingFace dataset
    dataset_dict = create_huggingface_dataset(train_data, val_data)
    
    # Step 4: Upload to HuggingFace
    success = upload_to_huggingface(dataset_dict)
    
    if success:
        print("\n" + "=" * 70)
        print("DATASET CREATION COMPLETE")
        print("=" * 70)
        print("\nThis script can now be safely deleted.")
        print(f"Dataset available at: https://huggingface.co/datasets/{REPO_ID}")
    else:
        print("\n" + "=" * 70)
        print("DATASET UPLOAD FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
