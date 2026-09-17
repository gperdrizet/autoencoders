# Autoencoders Demo

Interactive demonstrations of autoencoder applications for AI/ML bootcamp students.

## Overview

This repository is a **survey of autoencoder applications** - the goal is to show
*what autoencoders can do*, not to dive deep into implementation details.

Two demonstrations focusing on image processing:

1. **Image compression** - Compress 256×256 images to 2048 numbers (96× ratio) using a convolutional AE trained on DF2K_OST high-quality photographs
2. **Image denoising** - Remove Gaussian noise from images; the AE learns the manifold of clean images and pushes noisy inputs back onto it
3. **Anomaly detection** - Use reconstruction error as an unsupervised anomaly detection technique for images that 'don't belong'

Each demo includes:
- **Training notebooks** - Step-by-step training with detailed explanations
- **Pre-trained models** - Ready-to-use models automatically downloaded from HuggingFace

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Pre-trained models and datasets are hosted on HuggingFace and download automatically on first use. No configuration required for students.


### 3. Run the training notebooks

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `notebooks/01-compression.ipynb` | Train compression autoencoder (latent_dim=2048, 256×256 images) |
| `notebooks/02-denoising.ipynb` | Train denoising autoencoder (σ=25 Gaussian noise, 256×256 images) |
| `notebooks/03-anomaly-detection-activity.ipynb` | Design and train your own anomaly detection autoencoder on CIFAR10 images |
| `notebooks/04-anomaly-detection-activity-solution.ipynb` | Activity solution |

Set `TRAIN_MODEL = False` in any notebook to skip training and use the pre-trained model from the in class demo.

---

## Dataset

### DF2K_OST 
Combined high-quality image dataset created for this demonstration repository (see `scripts/build_df2k_ost_dataset.py`)

- Images from DIV2K, Flickr2K, and OST datasets
- ~30k high-quality images resized to 256×256 with Lanczos resampling
- 90/10 train/validation split
- Hosted on HuggingFace: [gperdrizet/DF2K_OST](https://huggingface.co/datasets/gperdrizet/DF2K_OST)
- Downloaded automatically on first run

### Original data sources

**DIV2K**
- **Citation**: Agustsson, E., & Timofte, R. (2017). NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.
- **Paper**: CVPR Workshops 2017
- **Website**: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Flickr2K**
- **Citation**: Agustsson, E. & Timofte, R. (2017). "NTIRE 2017 challenge on single image super-resolution: Dataset and study." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
- **Repository**: https://github.com/limbee/NTIRE2017

**OST (Outdoor Scene Training)**
- **Citation**: Wang, X., Yu, K., Dong, C., & Loy, C. C. (2018). Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform. arXiv:1804.02815
- **Paper**: https://arxiv.org/abs/1804.02815
- **Repository**: https://github.com/xinntao/SFTGAN

### Citation

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

## License

MIT License - see LICENSE file for details
