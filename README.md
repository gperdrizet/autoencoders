# Autoencoders demo

Interactive demonstrations of autoencoder applications for AI/ML bootcamp students.

## Overview

This repository contains three comprehensive demonstrations showing the power and versatility of autoencoders using the TF Flowers dataset:

1. **Image compression** - Compress images while maintaining quality
2. **Anomaly detection** - Identify unusual patterns using reconstruction error
3. **VAE generation** - Generate new synthetic images with Variational Autoencoders

Each demo includes:
- **Training notebooks** - Step-by-step training with detailed explanations
- **Interactive web app** - Streamlit-based demo for hands-on exploration
- **Quality metrics** - PSNR, SSIM, ROC-AUC, and more
- **Pre-trained models** - Ready-to-use models in `.keras` format

## Quick start

### Prerequisites

- Python 3.8+
- (Optional) NVIDIA GPU with CUDA for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd autoencoders
   ```

2. **Install dependencies**

   For local development with GPU:
   ```bash
   pip install -r requirements.txt
   ```

   For Streamlit Cloud deployment:
   ```bash
   pip install -r requirements-cloud.txt
   ```

3. **Configure Hugging Face (Optional)**
   
   Pre-trained models are hosted on Hugging Face and will download automatically. To use your own models:
   
   ```bash
   cp .env.example .env
   # Edit .env and set your HF_REPO_ID
   ```

### Running the Demos

#### Option 1: Streamlit Web App (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

**Note:** Pre-trained models will automatically download from Hugging Face on first use. This may take a few minutes depending on your connection.

#### Option 2: Training Notebooks

Open Jupyter and run the training notebooks in order:

```bash
jupyter notebook
```

Navigate to:
1. `notebooks/01-compression.ipynb` - Train compression autoencoders
2. `notebooks/02-anomaly_detection.ipynb` - Train anomaly detector
3. `notebooks/03-generation.ipynb` - Train VAE for generation

## ## Project structure

```
autoencoders/
├── app.py                          # Main Streamlit app (landing page)
├── pages/                          # Streamlit demo pages
│   ├── 01-compression.py           # Compression demo
│   ├── 02-anomaly_detection.py     # Anomaly detection demo
│   └── 03-vae_generation.py        # VAE generation demo
├── notebooks/                      # Training notebooks
│   ├── 01-compression.ipynb        # Train compression models
│   ├── 02-anomaly_detection.ipynb  # Train anomaly detector
│   └── 03-generation.ipynb         # Train VAE
├── src/                           # Shared utilities
│   ├── __init__.py
│   ├── model_utils.py            # Model architectures & loading
│   ├── data_utils.py             # TF Flowers loading & preprocessing
│   ├── visualization.py          # Plotting functions
│   ├── metrics.py                # Quality metrics
│   ├── vae_navigation.py         # VAE interactive components
│   ├── streamlit_components.py   # Reusable UI components
│   └── huggingface_utils.py      # Model download utilities
├── models/                        # Saved models (auto-downloaded from HF)
│   ├── compression_ae_latent32.keras
│   ├── compression_ae_latent64.keras
│   ├── compression_ae_latent128.keras
│   ├── compression_ae_latent256.keras
│   ├── anomaly_ae.keras
│   └── vae.keras
├── data/                          # TF Flowers cache (auto-downloaded)
├── logs/                          # TensorBoard logs
├── .streamlit/                    # Streamlit configuration
│   └── config.toml
├── upload_models.py               # Script to upload models to HF
├── requirements.txt               # Local development dependencies
├── requirements-cloud.txt         # Cloud deployment dependencies
├── .env.example                   # Environment configuration template
└── README.md                      # This file
```

## Demos

### 1. Image compression

**What it does:**
- Compresses 32x32 RGB images into smaller latent representations
- Reconstructs images from compressed form
- Compares quality across different compression ratios

**Key Features:**
- Multiple compression levels (32, 64, 128, 256 latent dimensions)
- Quality metrics (MSE, PSNR, SSIM)
- Upload your own images
- Side-by-side comparison
- Difference heatmaps

**Compression Ratios:**
- Latent 32: ~96x compression
- Latent 64: ~48x compression
- Latent 128: ~24x compression
- Latent 256: ~12x compression

### 2. Anomaly detection

**What it does:**
- Detects unusual/anomalous patterns in images
- Uses reconstruction error as anomaly score
- Trained on 4 flower classes, detects roses as anomalies

**Key Features:**
- Adjustable detection threshold
- ROC curve analysis
- Confusion matrix
- Top anomalies visualization
- Error distribution histograms
- Per-image error heatmaps

**Performance:**
- Achieves ROC-AUC > 0.9
- Configurable precision/recall trade-off
- Real-time threshold adjustment

### 3. VAE generation

**What it does:**
- Generates brand new, synthetic images
- Explores smooth probabilistic latent space
- Performs latent space arithmetic

**Key Features:**
- Random image generation
- Manual latent space navigation (sliders)
- Smooth interpolation between images
- Latent arithmetic (A + B - C)
- Temperature-controlled sampling

**What's Special:**
- Variational approach with KL divergence
- Reparameterization trick for gradients
- Continuous, structured latent space

## Training the models

**Note:** Pre-trained models are available on [Hugging Face Hub](https://huggingface.co/your-username/autoencoders-demo) and will download automatically when you run the app. You only need to train models if you want to experiment with different architectures or hyperparameters.

### Step 1: Compression Autoencoders

Run `notebooks/01-compression.ipynb` to train models with different latent dimensions:

- Trains 4 models (latent dims: 32, 64, 128, 256)
- ~50 epochs per model
- Saves to `models/compression_ae_latent{dim}.keras`
- Includes quality analysis and visualizations

**Training Time:** ~30-60 minutes on GPU, ~2-4 hours on CPU

### Step 2: Anomaly Detection

Run `notebooks/02-anomaly_detection.ipynb` to train the anomaly detector:

- Trains on 4 flower classes (dandelion, daisy, tulips, sunflowers)
- Uses roses as anomalies
- ~50 epochs
- Saves to `models/anomaly_ae.keras`
- Includes ROC analysis and threshold optimization

**Training Time:** ~20-40 minutes on GPU, ~1-2 hours on CPU

### Step 3: Variational Autoencoder

Run `notebooks/03-generation.ipynb` to train the VAE:

- Trains on all flower classes
- ~100 epochs (VAEs need more training)
- Saves to `models/vae.keras`
- Includes interpolation and latent space visualization

**Training Time:** ~60-90 minutes on GPU, ~4-6 hours on CPU

### Uploading Models to Hugging Face

After training your models, you can upload them to Hugging Face:

1. **Create a Hugging Face account**
   - Sign up at [huggingface.co](https://huggingface.co)
   - Create a new model repository

2. **Login via CLI**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

3. **Upload your models**
   ```bash
   python upload_models.py --repo-id your-username/autoencoders-demo
   ```

4. **Update configuration**
   - Edit `src/huggingface_utils.py`
   - Set `HF_REPO_ID = "your-username/autoencoders-demo"`
   - Commit and push changes

## Using the Streamlit app

### Local Deployment

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Cloud Deployment (Streamlit Community Cloud)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add autoencoder demos"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Set Python version: 3.9+
   - Click "Deploy"

3. **Configuration**
   - Uses `requirements-cloud.txt` automatically
   - TensorFlow CPU version for smaller footprint
   - Models download automatically from Hugging Face
   - Set `HF_REPO_ID` in Streamlit Cloud secrets if using a private repository

## Model details

### Architecture

All models use **4-5 layer convolutional architectures**:

**Encoder:**
```
Input (32x32x3)
  | Conv2D(64) + BN + ReLU
  | Conv2D(128) + BN + ReLU
  | Conv2D(256) + BN + ReLU
  | Conv2D(512) + BN + ReLU
  | Flatten -> Dense(latent_dim)
Latent Vector
```

**Decoder:**
```
Latent Vector
  | Dense(2x2x512) -> Reshape
  | Conv2DTranspose(512) + BN + ReLU
  | Conv2DTranspose(256) + BN + ReLU
  | Conv2DTranspose(128) + BN + ReLU
  | Conv2DTranspose(64) + BN + ReLU
  | Conv2D(3) + Sigmoid
Output (32x32x3)
```

**VAE Differences:**
- Encoder outputs `z_mean` and `z_log_var`
- Sampling layer with reparameterization trick
- Loss = Reconstruction + beta*KL Divergence (beta=0.0005)

### Training Configuration

- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE (compression/anomaly), MSE+KL (VAE)
- **Batch Size:** 128
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

## ## Development

### Running Tests

```bash
# TODO: Add tests
pytest tests/
```

### Code Style

The codebase follows:
- PEP 8 style guidelines
- Type hints for function parameters
- Comprehensive docstrings
- Modular, reusable components

### Project Philosophy

- **Educational Focus:** Clear explanations and visualizations
- **Modularity:** Shared utilities for both notebooks and Streamlit
- **Reproducibility:** Fixed random seeds, saved models
- **Deployment-Ready:** Optimized for both local and cloud

## ## Learning resources

### Key Concepts

- **Autoencoders:** Neural networks that learn compressed representations
- **Latent Space:** The compressed representation (bottleneck)
- **Reconstruction Error:** Difference between input and output
- **Variational Autoencoders:** Probabilistic generative models
- **KL Divergence:** Regularization term for VAEs

### Further Reading

**Autoencoders & VAEs:**
- [Original VAE Paper](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al., 2017
- [Understanding VAEs](https://arxiv.org/abs/1606.05908) - Doersch, 2016
- [Anomaly Detection Survey](https://arxiv.org/abs/2007.02500)

**Hugging Face Resources:**
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Model Hosting Guide](https://huggingface.co/docs/hub/models)
- [Getting Started with Hub](https://huggingface.co/docs/huggingface_hub/quick-start)

## ## Contributing

This is an educational project for bootcamp students. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ## License

MIT License - see LICENSE file for details

## ## Acknowledgments

- TF Flowers dataset from TensorFlow Datasets
- CIFAR-10 dataset (original base for this repo): [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- TensorFlow/Keras team
- Streamlit team
- AI/ML Bootcamp students and instructors

## ## Support

For questions or issues:
- Open an issue on GitHub
- Contact the bootcamp instructors
- Check the documentation in notebooks and code comments

---

**Built for AI/ML bootcamp students**