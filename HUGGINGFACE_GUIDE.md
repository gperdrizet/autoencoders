# Hugging Face Integration - Quick Guide

## Overview

This project uses **Hugging Face Hub** to store and distribute pre-trained models. This means:

- **No Git LFS complexity** - Models aren't in the repository  
- **Automatic downloads** - Models download when needed  
- **Easy updates** - Just upload new versions to Hugging Face  
- **Free hosting** - Hugging Face provides free model hosting  

## For Students (Using Pre-trained Models)

### Quick Start

Just clone and run - that's it! Models download automatically:

```bash
git clone <repository-url>
cd autoencoders
pip install -r requirements.txt
streamlit run app.py
```

When you first run the app or notebooks, models will download from Hugging Face. This happens only once - they're cached locally afterward.

### What Happens Behind the Scenes

1. You run the app or notebook
2. Code checks if model exists locally
3. If not found, downloads from Hugging Face
4. Saves to `models/` directory
5. Uses local copy on subsequent runs

### Troubleshooting

**Problem:** Download fails  
**Solution:** Check your internet connection. Models are 50-200MB each.

**Problem:** "Model not found" error  
**Solution:** Make sure `HF_REPO_ID` is set correctly in `src/huggingface_utils.py`

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

### Training and Uploading Models

1. **Train models** using the notebooks:
   ```bash
   # Run all cells in:
   # - notebooks/01-compression.ipynb
   # - notebooks/02-anomaly_detection.ipynb
   # - notebooks/03-generation.ipynb
   ```

2. **Upload to Hugging Face**:
   ```bash
   python upload_models.py --repo-id your-username/autoencoders-demo
   ```

3. **Update configuration**:
   - Edit `src/huggingface_utils.py`
   - Change: `HF_REPO_ID = "your-username/autoencoders-demo"`
   - Commit and push

4. **Share with students**:
   ```bash
   git add .
   git commit -m "Configure Hugging Face model downloads"
   git push
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
   ```

4. **Update code** to use token:
   ```python
   # In src/huggingface_utils.py
   import os
   HF_TOKEN = os.getenv("HF_TOKEN")
   
   # In download_model function:
   downloaded_path = hf_hub_download(
       repo_id=HF_REPO_ID,
       filename=model_name,
       token=HF_TOKEN,  # Add this line
       # ... rest of parameters
   )
   ```

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
