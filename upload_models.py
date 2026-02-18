#!/usr/bin/env python3
"""
Upload Trained Models to Hugging Face Hub

This script helps you upload your trained autoencoder models to Hugging Face Hub.
This makes it easy for students to download pre-trained models without Git LFS.

Usage:
    python upload_models.py

Prerequisites:
    1. Create a Hugging Face account at https://huggingface.co
    2. Create a model repository at https://huggingface.co/new
    3. Install huggingface-hub: pip install huggingface-hub
    4. Login: huggingface-cli login
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
import argparse


def upload_models(repo_id, models_dir='models', private=False):
    """
    Upload all trained model files to Hugging Face Hub.
    
    Args:
        repo_id (str): Hugging Face repository ID (e.g., 'username/autoencoders-demo')
        models_dir (str): Directory containing trained models
        private (bool): Whether the repository should be private
    """
    api = HfApi()
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f'ERROR: Models directory not found: {models_path}')
        return
    
    # Find all .keras model files
    model_files = list(models_path.glob('*.keras'))
    
    if not model_files:
        print(f'ERROR: No .keras model files found in {models_path}')
        print('   Please train the models first using the notebooks.')
        return
    
    print(f'\nUploading {len(model_files)} models to Hugging Face Hub')
    print(f'   Repository: {repo_id}')
    print(f'   Models directory: {models_path.absolute()}\n')
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type='model', private=private, exist_ok=True)
        print(f'SUCCESS: Repository ready: https://huggingface.co/{repo_id}\n')
    except Exception as e:
        print(f'ERROR: Failed to create repository: {e}')
        return
    
    # Upload each model file
    uploaded = []
    failed = []
    
    for model_file in model_files:
        try:
            print(f'Uploading {model_file.name}...')
            
            upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=model_file.name,
                repo_id=repo_id,
                repo_type='model',
            )
            
            uploaded.append(model_file.name)
            print(f'   Success!\n')
            
        except Exception as e:
            failed.append(model_file.name)
            print(f'   Failed: {e}\n')
    
    # Summary
    print('\n' + '='*60)
    print('UPLOAD SUMMARY')
    print('='*60)
    print(f'Uploaded: {len(uploaded)}/{len(model_files)} models')
    
    if uploaded:
        print('\nSuccessfully uploaded:')
        for name in uploaded:
            print(f'  - {name}')
    
    if failed:
        print('\nFailed to upload:')
        for name in failed:
            print(f'  - {name}')
    
    print('\n' + '='*60)
    print(f'\nModels are now available at:')
    print(f'   https://huggingface.co/{repo_id}\n')
    print('Next steps:')
    print(f'1. Update src/huggingface_utils.py with your repo ID:')
    print(f'   HF_REPO_ID = \'{repo_id}\'')
    print('2. Commit and push your code to GitHub')
    print('3. Students can now clone and run without training models!')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Upload trained models to Hugging Face Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_models.py --repo-id username/autoencoders-demo
  python upload_models.py --repo-id myorg/autoencoders --private
  python upload_models.py --repo-id username/models --models-dir ./my_models

Prerequisites:
  1. Create HF account: https://huggingface.co
  2. Login: huggingface-cli login
  3. Models must be trained first (run the notebooks)
        """
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        help='Hugging Face repository ID (e.g., username/autoencoders-demo)',
        required=True
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    
    args = parser.parse_args()
    
    # Check if user is logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f'\nLogged in as: {user["name"]}')
    except Exception:
        print('\nERROR: Not logged in to Hugging Face')
        print('   Please run: huggingface-cli login')
        return
    
    upload_models(args.repo_id, args.models_dir, args.private)


if __name__ == '__main__':
    main()
