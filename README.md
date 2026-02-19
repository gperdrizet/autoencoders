# Autoencoders demo

Interactive demonstrations of autoencoder applications for AI/ML bootcamp students.

## Overview

This repository contains three comprehensive demonstrations showing the power and versatility of autoencoders:

1. **Image compression** - Compress COCO images while maintaining quality
2. **Image denoising** - Remove noise from corrupted images
3. **Anomaly detection** - Identify unusual patterns using reconstruction error (Flowers vs COCO)

Each demo includes:
- **Training notebooks** - Step-by-step training with detailed explanations
- **Interactive web app** - Streamlit-based demo for hands-on exploration
- **Pre-trained models** - Ready-to-use models automatically downloaded from HuggingFace

## Quick start

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

3. **Configure environment**
   
   Pre-trained models and datasets are hosted on HuggingFace and will download automatically.
   
   ```bash
   cp .env.example .env
   ```
   
   **For students:** Leave `.env` as-is to use pre-trained models - no HuggingFace account needed!
   
   **For instructors:** To re-train and upload your own models:
   - Create a HuggingFace repository at https://huggingface.co/new
   - Edit `.env` and update `HF_REPO_ID` with your repo name
   - Get a token from https://huggingface.co/settings/tokens (write access)
   - Add `HF_TOKEN` to `.env`
   - Set `TRAIN_MODEL=True` in notebooks - trained models will auto-upload

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
1. `notebooks/01-compression.ipynb` - Train compression autoencoders (or download pre-trained)
2. `notebooks/02-anomaly_detection.ipynb` - Train anomaly detector (or download pre-trained)
3. `notebooks/02-denoising.ipynb` - Train denoising autoencoder (or download pre-trained)

**Tip:** Set `TRAIN_MODEL = False` at the top of each notebook to use pre-trained models without training.

## Project structure

```
autoencoders/
├── app.py                          # Main Streamlit app (landing page)
├── pages/                          # Streamlit demo pages
│   ├── 01-compression.py           # Compression demo
│   ├── 02-anomaly_detection.py     # Anomaly detection demo
│   └── 03-denoising.py             # Denoising demo
├── notebooks/                      # Training notebooks
│   ├── 01-compression.ipynb        # Train compression models
│   ├── 02-anomaly_detection.ipynb  # Train anomaly detector
│   └── 02-denoising.ipynb          # Train denoising model
├── src/                           # Shared utilities
│   ├── __init__.py
│   ├── model_utils.py            # Model architectures & loading
│   ├── data_utils.py             # Dataset loading & preprocessing
│   ├── visualization.py          # Plotting functions
│   ├── metrics.py                # Quality metrics
│   ├── streamlit_components.py   # Reusable UI components
│   └── huggingface_utils.py      # Model/dataset download/upload
├── models/                        # Saved models (auto-downloaded from HF)
│   ├── compression_ae_latent32.keras
│   ├── compression_ae_latent64.keras
│   ├── compression_ae_latent128.keras
│   ├── compression_ae_latent256.keras
│   ├── anomaly_ae.keras
│   └── denoising_ae.keras
├── data/                          # Dataset cache (auto-downloaded)
├── logs/                          # TensorBoard logs
├── .streamlit/                    # Streamlit configuration
│   └── config.toml
├── requirements.txt               # Local development dependencies
├── requirements-cloud.txt         # Cloud deployment dependencies
├── .env.example                   # Environment configuration template
└── README.md                      # This file
```


## Training the models

**Note:** Pre-trained models are available on [Hugging Face Hub](https://huggingface.co/gperdrizet/autoencoders) and will download automatically when you run the app. You only need to train models if you want to experiment with different architectures or hyperparameters.

### Compression

Run `notebooks/01-compression.ipynb` to train models with different latent dimensions:

- Trains 4 models (latent dims: 32, 64, 128, 256)
- ~50 epochs per model
- Saves to `models/compression_ae_latent{dim}.keras`
- Includes quality analysis and visualizations

**Training Time:** ~30-60 minutes on GPU, ~2-4 hours on CPU

### Anomaly detection

Run `notebooks/03-anomaly_detection.ipynb` to train the anomaly detector:

- Trains on 4 flower classes (dandelion, daisy, tulips, sunflowers)
- Uses roses as anomalies
- ~50 epochs
- Saves to `models/anomaly_ae.keras`
- Includes ROC analysis and threshold optimization

**Training Time:** ~20-40 minutes on GPU, ~1-2 hours on CPU


## Using the Streamlit app

### Local Deployment

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Cloud deployment (Streamlit Community Cloud)

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

## Contributing

This is an educational project for bootcamp students. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details
