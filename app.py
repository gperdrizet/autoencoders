"""
Autoencoder Demonstrations - Interactive Web App

Welcome to the autoencoder demo app! This application provides interactive
demonstrations of three key autoencoder applications.
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
of autoencoders using real-world datasets.

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
    ### Image denoising
    
    **What it does:**
    - Removes noise from corrupted images
    - Restores clean images from noisy inputs
    
    **Key Features:**
    - Multiple noise types (Gaussian, Salt & Pepper, Speckle)
    - Adjustable noise levels
    - Quality improvement metrics
    - Visual comparisons
    
    **Use Cases:**
    - Photo restoration
    - Medical image enhancement
    - Satellite image processing
    - Old photo restoration
    """)

st.markdown('---')

# Technical details
with st.expander('Technical details'):
    st.markdown("""
    ### About the Datasets
    
    **COCO 2017** (for Compression & Denoising):
    - 10% subset (~11,800 images)
    - 80 diverse object categories
    - Images resized to 64x64 RGB
    - Rich variety: people, vehicles, animals, objects, scenes
    
    **TF Flowers** (for Anomaly Detection):
    - 5 classes: dandelion, daisy, tulips, sunflowers, roses
    - ~3,700 color photos of flowers
    - Images resized to 64x64 RGB
    - Used as 'normal' data; COCO images as 'anomalies'
    
    ### Architecture
    
    - **Compression & Denoising**: Convolutional Autoencoders
        - 4-5 convolutional layers in encoder and decoder
        - Various latent dimensions (32, 64, 128, 256) for compression
        - BatchNormalization for stable training
    
    - **Anomaly Detection**: Trained only on flowers
        - Uses reconstruction error for detection
        - Higher error = likely anomaly (non-flower)
    
    ### Training
    
    - Framework: TensorFlow/Keras
    - Optimizer: Adam
    - Loss: MSE (all models)
    - Epochs: 20-50 depending on model
    - GPU accelerated training
    
    ### Source Code
    
    All source code and training notebooks are available in the GitHub repository.
    Models are saved in `.keras` format and can be downloaded from Hugging Face.
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
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9em;'>\n    <p>Built with Streamlit | Powered by TensorFlow/Keras</p>\n    <p>For educational purposes | AI/ML Bootcamp</p>\n</div>\n", unsafe_allow_html=True)

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
    
    1. **Compression** - Reduce image size (COCO dataset)
    2. **Anomaly Detection** - Find outliers (flowers vs non-flowers)
    3. **Denoising** - Remove noise from images (COCO dataset)
    
    Models trained on 64x64 RGB images.
    """)
