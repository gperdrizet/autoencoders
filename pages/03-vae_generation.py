"""
VAE Generation Demo - Streamlit Page

Interactive demonstration of image generation using Variational Autoencoders.
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Local imports
from src.data_utils import CIFAR10_CLASSES, load_cifar10
from src.model_utils import load_model
from src.streamlit_components import (
    render_header,
    render_model_info_sidebar,
    render_explanation_expander,
    create_download_button,
    show_loading_message
)
from src.vae_navigation import (
    render_latent_sliders,
    render_interpolation_controls,
    generate_from_latent,
    render_latent_arithmetic
)

# Page config
st.set_page_config(
    page_title='VAE Generation',
    page_icon='⬛',
    layout='wide'
)

# Header
render_header(
    'Image generation with variational autoencoders',
    'Explore the latent space and generate new synthetic images'
)

# Explanation
render_explanation_expander(
    'How do VAEs generate images?',
    """
    Variational Autoencoders (VAEs) are a type of generative model that can create new images:
    
    **Key Differences from Standard Autoencoders:**
    
    | Feature | Standard AE | VAE |
    |---------|-------------|-----|
    | Latent Space | Deterministic points | Probabilistic distributions |
    | Encoding | Direct mapping | Mean + Variance |
    | Loss | Reconstruction only | Reconstruction + KL Divergence |
    | Primary Use | Compression, Denoising | **Generation** |
    
    **How It Works:**
    
    1. **Encoder**: Maps images to a probability distribution (μ, σ²)
    2. **Sampling**: Sample a point from this distribution using the reparameterization trick
    3. **Decoder**: Generate an image from the sampled point
    
    **The Magic of VAEs:**
    - The latent space is **continuous and smooth**
    - Points close together → similar images
    - We can **interpolate** between images
    - We can sample **random points** to generate new images
    
    **KL Divergence:**
    - Regularizes the latent space to follow N(0,1)
    - Ensures the latent space is well-structured
    - Enables random sampling for generation
    
    **In this demo, you can:**
    - Generate random images by sampling the latent space
    - Navigate the latent space with sliders
    - Interpolate smoothly between two images
    - Perform arithmetic in latent space
    """
)

st.markdown('---')

# Load model
@st.cache_resource
def load_vae_model():
    model_path = Path(__file__).parent.parent / 'models' / 'vae.keras'
    if not model_path.exists():
        st.error(f'Model not found: {model_path}')
        st.info('Please train the VAE model first by running the generation notebook.')
        st.stop()
    
    # VAE needs custom objects
    from src.model_utils import Sampling
    model = load_model(str(model_path))
    
    # Extract encoder and decoder
    encoder = model.encoder
    decoder = model.decoder
    
    return model, encoder, decoder

with show_loading_message('Loading VAE model...'):
    vae, encoder, decoder = load_vae_model()

# Show model info
render_model_info_sidebar(vae, 'Variational Autoencoder')

# Get latent dimension from encoder output
latent_dim = encoder.output[0].shape[-1]  # z_mean shape

st.sidebar.markdown('---')
st.sidebar.markdown(f'**Latent Dimension**: {latent_dim}')

# Load dataset
@st.cache_data
def load_dataset():
    (x_train, y_train), (x_test, y_test) = load_cifar10(normalize=True)
    return x_test, y_test

x_test, y_test = load_dataset()

# Main content - Navigation mode selection
st.markdown('## Generation mode')

mode = st.radio(
    'Choose how to generate images:',
    ['Random generation', 'Manual navigation', 'Interpolation', 'Latent arithmetic'],
    horizontal=True
)

st.markdown('---')

if mode == 'Random generation':
    st.markdown('## Random image generation')
    st.markdown('Sample random points from the latent space to generate new images')
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('### Controls')
        
        n_images = st.slider(
            'Number of images to generate:',
            min_value=1,
            max_value=20,
            value=9,
            step=1
        )
        
        temperature = st.slider(
            'Sampling temperature:',
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help='Higher = more diversity, Lower = more typical samples'
        )
        
        if st.button('Generate new images', type='primary'):
            st.session_state['generate_random'] = True
    
    with col2:
        st.markdown('### Generated Images')
        
        if 'generate_random' in st.session_state or 'generated_images' not in st.session_state:
            with show_loading_message('Generating images...'):
                # Sample from latent space
                random_z = np.random.randn(n_images, latent_dim) * temperature
                
                # Generate images
                generated = decoder.predict(random_z, verbose=0)
                
                st.session_state['generated_images'] = generated
        
        # Display images
        generated = st.session_state['generated_images']
        
        # Calculate grid dimensions
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                img_idx = row * n_cols + col_idx
                if img_idx < len(generated):
                    with cols[col_idx]:
                        st.image(generated[img_idx], use_container_width=True)
                        create_download_button(
                            generated[img_idx],
                            filename=f'generated_{img_idx}.png',
                            button_text=f'Download {img_idx+1}'
                        )
    
    with st.expander('About random generation'):
        st.markdown("""
        **How it works:**
        - Sample random vectors from N(0, σ²) where σ = temperature
        - Pass through the decoder to generate images
        
        **Temperature parameter:**
        - **Low (0.5)**: Samples close to mean → more "average" images
        - **Medium (1.0)**: Standard sampling → balanced diversity
        - **High (2.0)**: Wider sampling → more unusual/diverse images
        
        **What to expect:**
        - Some images will look realistic
        - Some may be blurry or blend multiple classes
        - Each generation is unique!
        """)

elif mode == 'Manual navigation':
    st.markdown('## Manual latent space navigation')
    st.markdown('Use sliders to manually navigate through the latent space')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render latent sliders
        latent_vector = render_latent_sliders(latent_dim, key_prefix='vae_manual')
    
    with col2:
        st.markdown('### Generated Image')
        
        # Generate image from latent vector
        with show_loading_message('Generating...'):
            generated_img = generate_from_latent(decoder, latent_vector)
        
        st.image(generated_img, use_container_width=True)
        
        create_download_button(
            generated_img,
            filename='manual_generated.png',
            button_text='Download image'
        )
        
        st.markdown('---')
        
        # Show latent vector stats
        with st.expander('Latent vector stats'):
            st.metric('Mean', f'{np.mean(latent_vector):.3f}')
            st.metric('Std Dev', f'{np.std(latent_vector):.3f}')
            st.metric('Min', f'{np.min(latent_vector):.3f}')
            st.metric('Max', f'{np.max(latent_vector):.3f}')
    
    st.markdown('---')
    
    with st.expander('Navigation tips'):
        st.markdown("""
        **How to use the sliders:**
        - Each slider controls one dimension of the latent space
        - Values typically range from -3 to +3 (3 standard deviations)
        - Small changes can have visible effects on the generated image
        
        **Preset buttons:**
        - **Random**: Sample a random latent vector
        - **Zero**: Set all dimensions to 0 (latent space center)
        - **Positive**: Set all dimensions to +1
        - **Negative**: Set all dimensions to -1
        
        **Exploration strategy:**
        - Start with random or zero
        - Adjust individual sliders to see their effect
        - Some dimensions may have more visible impact than others
        """)

elif mode == 'Interpolation':
    st.markdown('## Latent space interpolation')
    st.markdown('Smoothly morph between two images by interpolating in latent space')
    
    # Render interpolation controls
    result = render_interpolation_controls(
        encoder, decoder, x_test, y_test,
        class_names=CIFAR10_CLASSES,
        key_prefix='vae_interp'
    )
    
    with st.expander('About interpolation'):
        st.markdown("""
        **How it works:**
        1. Encode both images to get their latent representations
        2. Create intermediate points by linear interpolation: z = (1-α)·z₁ + α·z₂
        3. Decode each intermediate point to get interpolated images
        
        **Why is it smooth?**
        - VAEs learn a continuous latent space
        - KL divergence regularization ensures smoothness
        - Points close together in latent space → similar images
        
        **Applications:**
        - Morphing animations
        - Exploring the space between concepts
        - Understanding what the model has learned
        
        **Tips:**
        - Try different class combinations
        - More steps = smoother transition
        - Look for meaningful intermediate representations
        """)

else:  # Latent Arithmetic
    st.markdown('## Latent space arithmetic')
    st.markdown('Perform arithmetic operations in latent space: **A + B - C**')
    
    # Render arithmetic interface
    result_img = render_latent_arithmetic(
        encoder, decoder, x_test,
        key_prefix='vae_arith'
    )
    
    st.markdown('---')
    
    with st.expander('About latent arithmetic'):
        st.markdown("""
        **The Concept:**
        - Each image maps to a point (actually a distribution) in latent space
        - We can perform vector arithmetic on these points
        - The decoder maps the result back to an image
        
        **Example Ideas:**
        - Cat + Dog - Bird = ?
        - Airplane + Automobile - Ship = ?
        - Frog + Horse - Deer = ?
        
        **What does it show?**
        - The latent space encodes semantic features
        - Arithmetic operations can combine or subtract features
        - Results may blend characteristics of the input images
        
        **Why it works:**
        - VAEs learn structured, continuous representations
        - Similar concepts cluster together
        - Feature directions are (somewhat) consistent
        
        **Limitations:**
        - Results aren't always meaningful
        - CIFAR-10 is low resolution (32×32)
        - Small dataset → limited feature learning
        - Works better with larger, more diverse datasets
        """)

# Additional statistics and information
st.markdown('---')
st.markdown('## Model statistics')

col1, col2, col3, col4 = st.columns(4)

# Sample some test images and compute reconstruction stats
sample_size = 100
sample_indices = np.random.choice(len(x_test), sample_size, replace=False)
sample_images = x_test[sample_indices]

with show_loading_message('Computing statistics...'):
    # Get reconstructions
    z_mean, z_log_var, z = encoder.predict(sample_images, verbose=0)
    reconstructions = decoder.predict(z_mean, verbose=0)  # Use mean for reconstruction
    
    # Compute metrics
    from src.metrics import calculate_mse, calculate_psnr
    mse_values = calculate_mse(sample_images, reconstructions)
    psnr_values = calculate_psnr(sample_images, reconstructions)

col1.metric('Avg Reconstruction MSE', f'{np.mean(mse_values):.6f}')
col2.metric('Avg PSNR', f'{np.mean(psnr_values):.2f} dB')
col3.metric('Latent Dimension', latent_dim)
col4.metric('Compression Ratio', f'{3072/latent_dim:.1f}×')

# Reconstruction comparison
with st.expander('Reconstruction quality examples'):
    st.markdown('Compare original test images with VAE reconstructions')
    
    n_compare = 5
    compare_indices = np.random.choice(len(x_test), n_compare, replace=False)
    compare_images = x_test[compare_indices]
    
    z_mean_comp, _, _ = encoder.predict(compare_images, verbose=0)
    reconstructed_comp = decoder.predict(z_mean_comp, verbose=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Original Images**')
        for i in range(n_compare):
            st.image(compare_images[i], width=150)
    
    with col2:
        st.markdown('**VAE Reconstructions**')
        for i in range(n_compare):
            st.image(reconstructed_comp[i], width=150)

# Educational content
st.markdown('---')
with st.expander('Learn more about VAEs'):
    st.markdown("""
    ### Key Concepts
    
    **1. Probabilistic Encoding**
    - Standard AE: x → z (deterministic)
    - VAE: x → (μ, σ²) → z (probabilistic)
    - Enables sampling and generation
    
    **2. Reparameterization Trick**
    - Problem: Can't backpropagate through random sampling
    - Solution: z = μ + σ ⊙ ε, where ε ~ N(0,1)
    - Now gradients can flow through μ and σ
    
    **3. Loss Function**
    ```
    Total Loss = Reconstruction Loss + β × KL Divergence
    
    Reconstruction: How well can we reconstruct the input?
    KL Divergence: How close is q(z|x) to N(0,1)?
    ```
    
    **4. Beta-VAE**
    - Vary β to control the trade-off
    - Higher β: More regularization, better generation, blurrier reconstructions
    - Lower β: Better reconstruction, less structured latent space
    - This model uses β = 0.0005 for CIFAR-10
    
    ### Applications Beyond This Demo
    
    - **Text Generation**: VAEs for sentences and documents
    - **Music Generation**: Create new melodies
    - **Molecule Design**: Drug discovery
    - **Recommendation Systems**: Learn user preference distributions
    - **Anomaly Detection**: Probabilistic anomaly scores
    - **Semi-supervised Learning**: Leverage unlabeled data
    
    ### Comparison with Other Generative Models
    
    | Model | Pros | Cons |
    |-------|------|------|
    | **VAE** | Principled framework, stable training, fast sampling | Can produce blurry images |
    | **GAN** | Sharp, realistic images | Unstable training, mode collapse |
    | **Diffusion** | SOTA quality | Slow sampling (many steps) |
    | **Autoregressive** | Flexible, high quality | Very slow generation |
    
    ### Further Reading
    
    - Original VAE Paper: [Kingma & Welling, 2013]
    - Beta-VAE: [Higgins et al., 2017]
    - Understanding VAEs: [Doersch, 2016]
    """)

# Sidebar additional info
with st.sidebar:
    st.markdown('---')
    st.markdown('## Quick actions')
    
    if st.button('Reset all'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown('---')
    st.markdown('## Pro tips')
    st.info("""
    **For best results:**
    
    1. Start with random generation to get a feel for what the model can create
    
    2. Use manual navigation to explore specific directions
    
    3. Try interpolation to see smooth transitions
    
    4. Experiment with latent arithmetic for creative combinations
    """)
