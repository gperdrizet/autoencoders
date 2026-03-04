# Autoencoders Demo

Interactive demonstrations of autoencoder applications for AI/ML bootcamp students.

## Overview

This repository is a **survey of autoencoder applications** - the goal is to show
*what autoencoders can do*, not to dive deep into implementation details.

Two demonstrations focusing on image processing:

1. **Image compression** - Compress 256×256 images to 2048 numbers (96× ratio) using a convolutional AE trained on DF2K_OST high-quality photographs
2. **Image denoising** - Remove Gaussian noise from images; the AE learns the manifold of clean images and pushes noisy inputs back onto it

Each demo includes:
- **Training notebooks** - Step-by-step training with detailed explanations
- **Interactive web app** - Streamlit-based demo for hands-on exploration
- **Pre-trained models** - Ready-to-use models automatically downloaded from HuggingFace

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Pre-trained models and datasets are hosted on HuggingFace and download automatically on first use. No configuration required for students.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

### 4. (Optional) Run the training notebooks

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `notebooks/01-compression.ipynb` | Train compression autoencoder (latent_dim=2048, 256×256 images) |
| `notebooks/02-denoising.ipynb` | Train denoising autoencoder (σ=25 Gaussian noise, 256×256 images) |

Set `TRAIN_MODEL = False` in any notebook to skip training and use the pre-trained model.

---

## Dataset

**DF2K_OST** (combined high-quality image dataset)
- Combined dataset from DIV2K, Flickr2K, and OST datasets
- ~13,874 high-quality images resized to 256×256 with Lanczos resampling
- 90/10 train/validation split
- Hosted on HuggingFace: [gperdrizet/DF2K_OST](https://huggingface.co/datasets/gperdrizet/DF2K_OST)
- Downloaded automatically on first run

---

## Project Structure

```
autoencoders/
├── app.py                              # Streamlit landing page
├── pages/
│   ├── 01-compression.py               # Image compression demo
│   └── 02-denoising.py                 # Image denoising demo
├── notebooks/
│   ├── 01-compression.ipynb            # Compression training notebook
│   └── 02-denoising.ipynb              # Denoising training notebook
├── src/
│   ├── data_utils.py                   # DF2K_OST loading, train/val split, noise
│   ├── model_utils.py                  # AE architectures (compression, denoising)
│   └── metrics.py                      # PSNR, SSIM, MSE
├── models/                             # Saved .keras models (downloaded from HF)
├── logs/                               # TensorBoard logs & result images
├── data/                               # Dataset cache
├── requirements.txt                    # Local / GPU dependencies
├── requirements-cloud.txt              # Streamlit Cloud (CPU) dependencies
└── .env                                # HuggingFace credentials
```

---

## Architecture

### Image autoencoder (compression & denoising)

```
Encoder                          Decoder
──────────────────────           ──────────────────────────────
Input  256×256×3                 Dense(16×16×512) → Reshape
  Conv2D(64,  s=2) → 128×128×64  ConvT(512, s=2) → 32×32×512
  Conv2D(128, s=2) → 64×64×128   ConvT(256, s=2) → 64×64×256
  Conv2D(256, s=2) → 32×32×256   ConvT(128, s=2) → 128×128×128
  Conv2D(512, s=2) → 16×16×512   ConvT(64,  s=2) → 256×256×64
  Flatten → Dense(latent_dim)    Conv2D(3, sigmoid) → 256×256×3
```

Each Conv block uses BatchNorm + LeakyReLU(0.2).

- **Compression**: `latent_dim=2048` → 96× compression ratio
- **Denoising**: `latent_dim=2048` → same architecture, trained on noisy inputs

---

## Performance targets

**Note**: Visual quality is the primary evaluation criterion. Metrics serve as guides, not hard targets.

| Demo | Metric | Guide | Notes |
|---|---|---|---|
| Compression | PSNR | > 30 dB | At 96× compression ratio (196,608 → 2048) |
| Compression | SSIM | > 0.90 | Visual similarity to original |
| Denoising | PSNR | > 28 dB | vs noisy input at σ=25 |
| Denoising | SSIM | > 0.85 | Noise removal without over-smoothing |

---

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file: `app.py`, Python 3.10+
4. The app uses `requirements-cloud.txt` (CPU-only TensorFlow) automatically
5. Models and datasets download automatically from HuggingFace

---

## License

MIT License - see LICENSE file for details
