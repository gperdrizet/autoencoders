"""
Autoencoder Demonstrations - Interactive Web App

Welcome to the autoencoder demo app! This application provides interactive
demonstrations of three key autoencoder applications on the CIFAR-10 dataset.
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from src.streamlit_components import render_header


# Page configuration
st.set_page_config(
    page_title='Autoencoder Demos',
    page_icon='⬛',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Main header
render_header(
    'Autoencoder demonstrations',
    'Interactive demos showing the power and versatility of autoencoders'
)

# Introduction
st.markdown("""
## Welcome to the autoencoder demo app

This application provides hands-on, interactive demonstrations of **three key applications** 
of autoencoders using the CIFAR-10 dataset.

### What are Autoencoders?

Autoencoders are neural networks that learn to compress data into a lower-dimensional 
representation (encoding) and then reconstruct it back to the original form (decoding). 
They consist of two main components:

- **Encoder**: Compresses input into a latent (compressed) representation
- **Decoder**: Reconstructs output from the latent representation

### Explore the Demos

Use the sidebar to navigate between three different applications:
""")

# Demo cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Image compression
    
    **What it does:**
    - Compresses images to reduce storage/bandwidth
    - Reconstructs images with minimal quality loss
    
    **Key Features:**
    - Compare different compression ratios
    - Measure quality metrics (PSNR, SSIM)
    - Upload your own images
    - Visualize compression artifacts
    
    **Use Cases:**
    - Image storage optimization
    - Bandwidth reduction
    - Feature extraction
    """)

with col2:
    st.markdown("""
    ### Anomaly detection
    
    **What it does:**
    - Identifies unusual/anomalous patterns
    - Uses reconstruction error as anomaly score
    
    **Key Features:**
    - Adjustable detection threshold
    - ROC curve analysis
    - Visual error heatmaps
    - Precision/recall tradeoffs
    
    **Use Cases:**
    - Fraud detection
    - Quality control
    - Network intrusion detection
    - Medical imaging
    """)

with col3:
    st.markdown("""
    ### VAE generation
    
    **What it does:**
    - Generates brand new, synthetic images
    - Explores smooth latent space
    
    **Key Features:**
    - Random image generation
    - Latent space navigation
    - Image interpolation
    - Latent arithmetic
    
    **Use Cases:**
    - Creative content generation
    - Data augmentation
    - Design exploration
    - Drug discovery
    """)

st.markdown('---')

# Technical details
with st.expander('Technical details'):
    st.markdown("""
    ### About the Models
    
    All models in this demo are trained on the **CIFAR-10 dataset**:
    - 60,000 32×32 color images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    ### Architecture
    
    - **Compression & Anomaly Detection**: Standard Convolutional Autoencoders
        - 4-5 convolutional layers in encoder and decoder
        - Various latent dimensions (32, 64, 128, 256)
        - BatchNormalization for stable training
    
    - **Generation**: Variational Autoencoder (VAE)
        - Probabilistic latent space
        - Reparameterization trick for backpropagation
        - KL divergence regularization
    
    ### Training
    
    - Framework: TensorFlow/Keras
    - Optimizer: Adam
    - Loss: MSE (compression/anomaly), MSE + KL divergence (VAE)
    - Epochs: 50-100 depending on model
    
    ### Source Code
    
    All source code and training notebooks are available in the GitHub repository.
    Models are saved in `.keras` format with float16 quantization for deployment.
    """)

# How to use
st.markdown('## Getting started')

st.markdown("""
1. **Choose a demo** from the sidebar navigation
2. **Interact** with the controls to explore different parameters
3. **Upload** your own images (for compression demo)
4. **Learn** from the visualizations and metrics

Each demo includes:
- Model information in the sidebar
- Interactive visualizations
- Quality metrics and performance indicators
- Download buttons for generated images
""")

# Footer
st.markdown('---')
st.markdown('<div style="text-align: center; color: #666; font-size: 0.9em;">\n    <p>Built with Streamlit • Powered by TensorFlow/Keras</p>\n    <p>For educational purposes • AI/ML Bootcamp</p>\n</div>\n', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown('## Navigation')
    st.info('Select a demo page from above to get started!')
    
    st.markdown('---')
    
    st.markdown('## Resources')
    st.markdown("""
    - [Training Notebooks](notebooks/)
    - [Source Code](src/)
    - [GitHub Repository](#)
    """)
    
    st.markdown('---')
    
    st.markdown('## About')
    st.markdown("""
    This app demonstrates three practical applications of autoencoders:
    
    1. **Compression** - Reduce image size
    2. **Anomaly Detection** - Find outliers
    3. **Generation** - Create new images
    
    All models are trained on CIFAR-10 (32×32 RGB images).
    """)
