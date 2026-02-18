# Hugging Face Integration - Quick Guide

## Overview

This project uses **Hugging Face Hub** to store and distribute:

- **Pre-trained models** - 6 trained autoencoder models
- **COCO dataset subset** - Pre-processed 10% subset (~300-500 MB)

This means:

- **No Git LFS complexity** - Models and data aren't in the repository  
- **Automatic downloads** - Everything downloads when needed  
- **No massive downloads** - Students get 300MB instead of 95GB for COCO
- **Easy updates** - Just upload new versions to Hugging Face  
- **Free hosting** - Hugging Face provides free hosting  

## For Students (Using Pre-trained Models & Data)

### Quick Start

Just clone and run - that's it! Models and data download automatically:

```bash
git clone <repository-url>
cd autoencoders
pip install -r requirements.txt
streamlit run app.py
```

When you first run the app or notebooks:
- **COCO dataset** downloads from Hugging Face (~300-500 MB, one-time)
- **Models** download when needed (~500 MB total)
- Everything caches locally for future use

### What Happens Behind the Scenes

**For Dataset Loading:**
1. Check local cache (`data/coco_10percent_subset.npz`)
2. If not found, download from Hugging Face (~300-500 MB)
3. If HF fails, fallback to TensorFlow Datasets (~95GB - rare)
4. Cache locally for next time

**For Models:**
1. Check if model exists locally (`models/`)
2. If not found, download from Hugging Face
3. Save to `models/` directory
4. Use local copy on subsequent runs

### Storage Requirements

- **With Hugging Face**: ~1 GB (models + dataset cache)
- **Without Hugging Face**: ~100 GB (full COCO download)

### Troubleshooting

**Problem:** Dataset download fails  
**Solution:** Check internet connection. If Hugging Face is down, it will fallback to TensorFlow Datasets (slow but reliable).

**Problem:** "Model not found" error  
**Solution:** Make sure `HF_REPO_ID` is set correctly in `src/huggingface_utils.py`

**Problem:** Download is slow
**Solution:** First-time setup downloads ~800 MB total. Subsequent runs use cached data.

## For Instructors (Uploading Models)

### Initial Setup

1. **Create Hugging Face account**
   ```bash
   # Sign up at https://huggingface.co
   ```

2. **Create model repository**
   - Go to https://huggingface.co/new
   - Choose "Model" type
   - Name it (e.g., "autoencoders-demo")
   - Public or Private (your choice)

3. **Login via CLI**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   # Paste your token from https://huggingface.co/settings/tokens
   ```

### Training and Uploading

1. **Upload the COCO dataset first** (one-time, instructor only):
   ```bash
   # This processes and uploads the 10% COCO subset
   python upload_dataset.py --repo-id your-username/autoencoders-demo
   ```
   
   **Note**: First run requires downloading full COCO (~95GB) from TensorFlow Datasets.
   After processing, only the 300-500 MB subset is uploaded to Hugging Face.
   This step is done once by the instructor - students just download the subset.

2. **Train models** using the notebooks:
   ```bash
   # Run all cells in:
   # - notebooks/01-compression.ipynb
   # - notebooks/02-denoising.ipynb  
   # - notebooks/02-anomaly_detection.ipynb
   ```

3. **Upload models to Hugging Face**:
   ```bash
   python upload_models.py --repo-id your-username/autoencoders-demo
   ```downloads"
   git push
   ```

### Expected Content on Hugging Face

After setup, your repository should have:

**Dataset:**
```
data/
└── coco_10percent_subset.npz       (~300-500 MB)
```

**Models:**
```
models/
├── compression_ae_latent32.keras   (~80MB)
├── compression_ae_latent64.keras   (~80MB)
├── compression_ae_latent128.keras  (~82MB)
├── compression_ae_latent256.keras  (~85MB)
├── anomaly_ae.keras                (~80MB)
└── denoising_ae.keras              (~80MB)
```

Total: ~800 MB - 1 GB (well within Hugging Face limits
   ```

### Expected Models

After training, you should have these files:

```
models/
├── compression_ae_latent32.keras   (~80MB)
├── compression_ae_latent64.keras   (~80MB)
├── compression_ae_latent128.keras  (~82MB)
├── compression_ae_latent256.keras  (~85MB)
├── anomaly_ae.keras                (~80MB)
└── vae.keras                       (~85MB)
```

Total: ~500MB (well within Hugging Face limits)

## Advanced: Private Repositories

If you want to keep models private:

1. **Make repository private** on Hugging Face

2. **Create access token**:
   - Go to https://huggingface.co/settings/tokens
   - Create "Read" token
   - Copy the token

3. **Configure for students**:
   
   Option A - Environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   streamlit run app.py
   ```
   
   Option B - Streamlit secrets (for cloud deployment):
   ```toml
   # .streamlit/secrets.toml
   HF_TOKEN = "your_token_here"
   HF_REPO_ID = "your-username/autoencoders-demo"
   ``No 95GB COCO download needed
   - Works on any machine with internet

2. **Better Learning Experience**
   - Students focus on ML, not data engineering
   - Introduces Hugging Face ecosystem
   - Industry-standard practices
   - Fast iteration and experimentation

3. **Easy Maintenance**
   - Update models without Git commits
   - Update dataset preprocessing independently
   - Version models and data separately
   - Track downloads and usage

4. **Cost-Effective**
   - Free Hugging Face hosting
   - Faster Git operations
   - Lower storage requirements
   - Reduced bandwidth for students

### Key Advantages: Dataset Hosting

**Without Hugging Face:**
- Students download 95GB from TensorFlow Datasets
- 30+ minutes download time
- Requires significant disk space
- Repeated downloads for multiple machines

**With Hugging Face:**
- Students download 300-500 MB preprocessed subset
- ~1 minute download time
- Minimal disk space needed
- Cached for reuse
- Guaranteed consistency (everyone uses same preprocessing)

## Benefits for Education

### Why This Approach?

1. **Simplicity for Students**
   - No complex Git LFS setup
   - No large files to clone
   - Works on any machine with internet

2. **Better Learning Experience**
   - Students focus on ML, not infrastructure
   - Introduces Hugging Face ecosystem
   - Industry-standard practices

3. **Easy Maintenance**
   - Update models without Git commits
   - Version models independently
   - Track downloads and usage

4. **Cost-Effective**
   - Free Hugging Face hosting
   - Faster Git operations
   - Lower storage requirements

### Learning Opportunities

This setup naturally introduces students to:

- Hugging Face Hub - industry-standard model sharing
- Model versioning and distribution
- Environment configuration (.env files)
- API usage and authentication
- Caching and optimization

## Monitoring and Management

### View Downloads

Visit your model repository page:
```
https://huggingface.co/your-username/autoencoders-demo
```

You can see:
- Total downloads
- Files and sizes
- Model card / README
- Community discussions

### Update Models

To update models after retraining:

```bash
python upload_models.py --repo-id your-username/autoencoders-demo
```

The script only uploads new/changed files.

### Delete Cache

Students can clear cached models:

```bash
rm -rf models/*.keras
rm -rf models/.cache
```

Models will re-download on next use.

## Support

For issues or questions:

- **Students**: Contact your instructor
- **Instructors**: 
  - [Hugging Face Documentation](https://huggingface.co/docs/hub)
  - [GitHub Issues](https://github.com/your-repo/issues)
  - Review `src/huggingface_utils.py` for implementation details

---

**Happy Teaching!**
