# Autoencoders Demo

Interactive demonstrations of autoencoder applications for AI/ML bootcamp students.

## Overview

This repository is a **survey of autoencoder applications** — the goal is to show
*what autoencoders can do*, not to dive deep into implementation details.

Three demos, three different domains:

1. **Image compression** — Compress 128×128 images to 128 numbers (384× ratio) using a convolutional AE trained on DF2K_OST high-quality photographs
2. **Image denoising** — Remove Gaussian noise from images; the AE learns the manifold of clean images and pushes noisy inputs back onto it
3. **ECG anomaly detection** — Train only on normal heartbeats; arrhythmias are flagged by their high reconstruction error — no anomaly labels needed for training

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

Pre-trained models and datasets are hosted on HuggingFace and download automatically.

```bash
cp .env.example .env
```

- **Students**: Leave `.env` as-is — no HuggingFace account needed.
- **Instructors**: Set `HF_TOKEN` (write access) and `HF_REPO_ID` to re-train and upload your own models.

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
| `notebooks/01-compression.ipynb` | Train compression autoencoder (latent_dim=128, 50 epochs) |
| `notebooks/02-denoising.ipynb` | Train denoising autoencoder (σ=25 Gaussian noise) |
| `notebooks/03-anomaly-detection.ipynb` | Train ECG anomaly detector (normal beats only) |

Set `TRAIN_MODEL = False` in any notebook to skip training and use the pre-trained model.

---

## Dataset

**DF2K_OST** (image demos)
- 900 high-quality photos from the DIV2K dataset, resized to 128×128 with Lanczos resampling
- Hosted on HuggingFace: [gperdrizet/autoencoders](https://huggingface.co/datasets/gperdrizet/autoencoders)
- Downloaded automatically on first run

**ECG5000** (anomaly detection)
- 5,000 single-heartbeat ECG windows, 140 time steps each
- Source: UCR Time Series Archive / TensorFlow datasets
- Classes: 1 = Normal, 2–5 = Various arrhythmias
- Downloaded from `storage.googleapis.com/tensorflow` on first run

---

## Project Structure

```
autoencoders/
├── app.py                              # Streamlit landing page
├── pages/
│   ├── 01-compression.py               # Image compression demo
│   ├── 02-denoising.py                 # Image denoising demo
│   └── 03-anomaly-detection.py         # ECG anomaly detection demo
├── notebooks/
│   ├── 01-compression.ipynb            # Compression training notebook
│   ├── 02-denoising.ipynb              # Denoising training notebook
│   └── 03-anomaly-detection.ipynb      # Anomaly detection training notebook
├── src/
│   ├── data_utils.py                   # DF2K_OST loading, train/val split, noise
│   ├── model_utils.py                  # AE architectures (compression, denoising, anomaly)
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

### Image Autoencoder (compression & denoising)

```
Encoder                          Decoder
──────────────────────           ──────────────────────────────
Input  128×128×3                 Dense(8×8×512) → Reshape
  Conv2D(64,  s=2) → 64×64×64    ConvT(512, s=2) → 16×16×512
  Conv2D(128, s=2) → 32×32×128   ConvT(256, s=2) → 32×32×256
  Conv2D(256, s=2) → 16×16×256   ConvT(128, s=2) → 64×64×128
  Conv2D(512, s=2) →  8×8×512    ConvT(64,  s=2) → 128×128×64
  Flatten → Dense(latent_dim)    Conv2D(3, sigmoid) → 128×128×3
```

Each Conv block uses BatchNorm + LeakyReLU(0.2).

- **Compression**: `latent_dim=128` → 384× compression ratio
- **Denoising**: `latent_dim=256` → better quality, less compression

### Anomaly Detection Autoencoder (dense / MLP)

```
Encoder: Input(140) → Dense(128) → Dense(64) → Dense(32)  [latent]
Decoder: Dense(64)  → Dense(128) → Dense(140, sigmoid)
```

---

## Performance Targets

| Demo | Metric | Target | Notes |
|---|---|---|---|
| Compression | PSNR | > 30 dB | At 384× compression ratio |
| Compression | SSIM | > 0.90 | |
| Denoising | PSNR | > 28 dB | vs noisy input at σ=25 |
| Denoising | SSIM | > 0.85 | |
| Anomaly Detection | AUC-ROC | > 0.90 | Normal vs all arrhythmia classes |

---

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file: `app.py`, Python 3.10+
4. Add `HF_REPO_ID` (and optionally `HF_TOKEN`) to Streamlit secrets
5. The app uses `requirements-cloud.txt` (CPU-only TensorFlow) automatically

---

## License


MIT License - see LICENSE file for details
