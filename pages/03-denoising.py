"""
Image Denoising Demo - Streamlit Page

Interactive demonstration of image denoising using autoencoders.
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
from src.data_utils import COCO_CLASSES, load_coco, preprocess_image
from src.metrics import compute_metrics_summary
from src.model_utils import load_model
from src.visualization import create_plotly_comparison, create_plotly_heatmap
from src.streamlit_components import (
    render_header,
    render_model_info_sidebar,
    render_image_uploader,
    render_sample_selector,
    create_download_button,
    render_metrics_display,
    render_explanation_expander,
    show_loading_message
)

# Page config
st.set_page_config(
    page_title='Denoising Demo',
    page_icon='⬛',
    layout='wide'
)

# Header
render_header(
    'Image denoising with autoencoders',
    'Remove noise and restore clean images using denoising autoencoders'
)

# Explanation
render_explanation_expander(
    'How does denoising work?',
    """
    A denoising autoencoder learns to remove noise from images:
    
    1. **Training**: The model learns from pairs of noisy and clean images
    2. **Encoding**: The encoder extracts robust features from noisy input
    3. **Decoding**: The decoder reconstructs the clean image
    
    **Noise Types**:
    - **Gaussian Noise**: Random pixel variations (common in low-light photos)
    - **Salt & Pepper**: Random black and white pixels (transmission errors)
    - **Speckle**: Multiplicative noise (radar/ultrasound images)
    
    **Quality Metrics**:
    - **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (>30 dB is good)
    - **SSIM** (Structural Similarity): Higher is better (0-1 scale, >0.9 is excellent)
    - Higher values indicate better noise removal
    
    **Applications**:
    - Photo restoration
    - Medical image enhancement
    - Satellite image processing
    - Old photo restoration
    """
)

st.markdown('---')

# Noise type selection
st.sidebar.markdown('## Configuration')

noise_type = st.sidebar.selectbox(
    'Noise Type:',
    ['Gaussian', 'Salt & Pepper', 'Speckle'],
    help='Type of noise to add to images'
)

noise_level = st.sidebar.slider(
    'Noise Level:',
    0.0, 0.5, 0.25, 0.05,
    help='Amount of noise to add (higher = more noise)'
)

# Load model
@st.cache_resource
def load_denoising_model():
    from src.huggingface_utils import download_model
    
    model_name = 'denoising_ae.keras'
    try:
        model_path = download_model(model_name, models_dir='models')
        return load_model(model_path)
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.info('Make sure the model is uploaded to Hugging Face or train it locally.')
        st.stop()

with show_loading_message('Loading denoising model...'):
    model = load_denoising_model()

# Show model info in sidebar
render_model_info_sidebar(model, 'Denoising AE')

# Load dataset
@st.cache_data
def load_dataset():
    (x_train, y_train), (x_test, y_test) = load_coco(subset_percent=10, normalize=True)
    return x_test, y_test

x_test, y_test = load_dataset()

# Noise functions
def add_gaussian_noise(images, noise_factor=0.3):
    """Add Gaussian noise to images."""
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy = images + noise
    return np.clip(noisy, 0.0, 1.0)

def add_salt_pepper_noise(images, amount=0.05):
    """Add salt and pepper noise to images."""
    noisy = images.copy()
    
    # Salt (white pixels)
    num_salt = int(amount * images.size * 0.5)
    coords_salt = [np.random.randint(0, i, num_salt) for i in images.shape]
    noisy[tuple(coords_salt)] = 1.0
    
    # Pepper (black pixels)
    num_pepper = int(amount * images.size * 0.5)
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in images.shape]
    noisy[tuple(coords_pepper)] = 0.0
    
    return noisy

def add_speckle_noise(images, noise_factor=0.2):
    """Add speckle (multiplicative) noise to images."""
    noise = np.random.randn(*images.shape) * noise_factor
    noisy = images + images * noise
    return np.clip(noisy, 0.0, 1.0)

# Main content
st.markdown('## Select an image')

# Tabs for different input methods
tab1, tab2 = st.tabs(['From dataset', 'Upload your own'])

with tab1:
    selected_image, selected_label, selected_idx = render_sample_selector(
        x_test, y_test, COCO_CLASSES, key='denoising'
    )
    input_image = selected_image

with tab2:
    uploaded_file = render_image_uploader(key='denoising_upload')
    if uploaded_file is not None:
        from src.data_utils import upload_and_preprocess
        input_image = upload_and_preprocess(uploaded_file)[0]
    else:
        st.info('Upload an image to see denoising results')
        input_image = None

# Process and display results
if input_image is not None:
    st.markdown('---')
    st.markdown('## Denoising results')
    
    # Prepare image
    clean_image = input_image if len(input_image.shape) == 3 else input_image.reshape(64, 64, 3)
    
    # Add noise based on selected type
    if noise_type == 'Gaussian':
        noisy_image = add_gaussian_noise(
            np.expand_dims(clean_image, axis=0), 
            noise_factor=noise_level
        )[0]
    elif noise_type == 'Salt & Pepper':
        noisy_image = add_salt_pepper_noise(
            np.expand_dims(clean_image, axis=0),
            amount=noise_level
        )[0]
    else:  # Speckle
        noisy_image = add_speckle_noise(
            np.expand_dims(clean_image, axis=0),
            noise_factor=noise_level
        )[0]
    
    # Denoise
    with show_loading_message('Denoising image...'):
        noisy_batch = np.expand_dims(noisy_image, axis=0)
        denoised = model.predict(noisy_batch, verbose=0)[0]
    
    # Display three images side by side
    st.markdown('### Comparison')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(clean_image, caption='Original (Clean)', use_container_width=True)
    
    with col2:
        st.image(noisy_image, caption=f'Noisy ({noise_type})', use_container_width=True)
    
    with col3:
        st.image(denoised, caption='Denoised', use_container_width=True)
    
    # Calculate metrics - compare denoised vs original
    from src.metrics import calculate_psnr, calculate_ssim
    
    psnr_noisy = calculate_psnr(clean_image, noisy_image)
    ssim_noisy = calculate_ssim(clean_image, noisy_image)
    psnr_denoised = calculate_psnr(clean_image, denoised)
    ssim_denoised = calculate_ssim(clean_image, denoised)
    
    # Display metrics
    st.markdown('### Quality metrics')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Noisy Image**')
        render_metrics_display({
            'PSNR': f'{psnr_noisy:.2f} dB',
            'SSIM': f'{ssim_noisy:.4f}'
        }, columns=2)
    
    with col2:
        st.markdown('**Denoised Image**')
        render_metrics_display({
            'PSNR': f'{psnr_denoised:.2f} dB',
            'SSIM': f'{ssim_denoised:.4f}'
        }, columns=2)
    
    # Show improvement
    psnr_improvement = psnr_denoised - psnr_noisy
    ssim_improvement = ssim_denoised - ssim_noisy
    
    st.markdown('### Improvement')
    render_metrics_display({
        'PSNR Improvement': f'+{psnr_improvement:.2f} dB' if psnr_improvement > 0 else f'{psnr_improvement:.2f} dB',
        'SSIM Improvement': f'+{ssim_improvement:.4f}' if ssim_improvement > 0 else f'{ssim_improvement:.4f}'
    }, columns=2)
    
    if psnr_improvement > 0 and ssim_improvement > 0:
        st.success('Denoising improved image quality!')
    elif psnr_improvement > 0 or ssim_improvement > 0:
        st.info('Denoising partially improved image quality.')
    else:
        st.warning('The model struggled with this noise level/type.')
    
    st.markdown('---')
    
    # Visualization options
    viz_mode = st.radio(
        'Visualization Mode:',
        ['Side-by-Side', 'Difference Heatmap (Noisy vs Clean)', 'Difference Heatmap (Denoised vs Clean)'],
        horizontal=True
    )
    
    if viz_mode == 'Side-by-Side':
        col1, col2 = st.columns(2)
        with col1:
            fig = create_plotly_comparison(
                clean_image,
                noisy_image,
                title=f'Original vs Noisy ({noise_type})'
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_plotly_comparison(
                clean_image,
                denoised,
                title='Original vs Denoised'
            )
            st.plotly_chart(fig, use_container_width=True)
    elif viz_mode == 'Difference Heatmap (Noisy vs Clean)':
        fig = create_plotly_heatmap(clean_image, noisy_image)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_plotly_heatmap(clean_image, denoised)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download buttons
    st.markdown('### Download results')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_download_button(
            clean_image,
            filename=f'original_{selected_idx}.png',
            button_text='Download Original'
        )
    
    with col2:
        create_download_button(
            noisy_image,
            filename=f'noisy_{noise_type.lower()}_{selected_idx}.png',
            button_text='Download Noisy'
        )
    
    with col3:
        create_download_button(
            denoised,
            filename=f'denoised_{selected_idx}.png',
            button_text='Download Denoised'
        )
    
    # Additional analysis
    with st.expander('Detailed analysis'):
        st.markdown(f"""
        **Denoising Performance:**
        - Original image quality: Clean reference
        - Noisy image quality: PSNR = {psnr_noisy:.2f} dB, SSIM = {ssim_noisy:.4f}
        - Denoised quality: PSNR = {psnr_denoised:.2f} dB, SSIM = {ssim_denoised:.4f}
        - Quality improvement: {'+' if psnr_improvement > 0 else ''}{psnr_improvement:.2f} dB PSNR
        
        **Interpretation:**
        - PSNR improvement of {psnr_improvement:.2f} dB indicates {'significant' if abs(psnr_improvement) > 5 else 'moderate' if abs(psnr_improvement) > 2 else 'slight'} quality change
        - SSIM improvement of {ssim_improvement:.4f} shows {'excellent' if ssim_improvement > 0.1 else 'good' if ssim_improvement > 0.05 else 'moderate'} structural restoration
        
        **Notes:**
        - The model was trained on Gaussian noise with factor 0.25
        - Performance may vary with different noise types or levels
        - Very high noise levels may be challenging to remove completely
        """)

# Comparison section
st.markdown('---')
st.markdown('## Compare noise types')

if st.button('Generate Comparison Across All Noise Types'):
    if input_image is not None:
        clean_image = input_image if len(input_image.shape) == 3 else input_image.reshape(64, 64, 3)
        
        st.markdown('### All noise types at current noise level')
        cols = st.columns(4)
        
        # Original
        with cols[0]:
            st.image(clean_image, caption='Clean', use_container_width=True)
            st.metric('Type', 'Original')
        
        # Each noise type
        noise_types = ['Gaussian', 'Salt & Pepper', 'Speckle']
        for idx, ntype in enumerate(noise_types, start=1):
            if ntype == 'Gaussian':
                noisy = add_gaussian_noise(np.expand_dims(clean_image, axis=0), noise_level)[0]
            elif ntype == 'Salt & Pepper':
                noisy = add_salt_pepper_noise(np.expand_dims(clean_image, axis=0), noise_level)[0]
            else:
                noisy = add_speckle_noise(np.expand_dims(clean_image, axis=0), noise_level)[0]
            
            with show_loading_message(f'Denoising {ntype}...'):
                denoised = model.predict(np.expand_dims(noisy, axis=0), verbose=0)[0]
                psnr = calculate_psnr(clean_image, denoised)
                ssim = calculate_ssim(clean_image, denoised)
            
            with cols[idx]:
                st.image(denoised, caption=f'{ntype}', use_container_width=True)
                st.metric('PSNR', f'{psnr:.1f} dB')
                st.metric('SSIM', f'{ssim:.3f}')
    else:
        st.warning('Please select or upload an image first!')

# Educational content
st.markdown('---')
with st.expander('Understanding different noise types'):
    st.markdown("""
    **Gaussian Noise**:
    - Most common type of noise in digital images
    - Caused by electronic sensor limitations, especially in low light
    - Each pixel gets a random value from a normal distribution
    - Appears as random graininess throughout the image
    - Example: Low-light smartphone photos, high ISO photography
    
    **Salt & Pepper Noise**:
    - Random pixels become pure white (salt) or pure black (pepper)
    - Caused by transmission errors, dead pixels, or analog-to-digital conversion errors
    - Appears as scattered white and black dots
    - Example: Old scanned documents, faulty camera sensors
    
    **Speckle Noise**:
    - Multiplicative noise (proportional to pixel intensity)
    - Common in coherent imaging systems
    - Appears as granular patterns
    - Example: Radar images, ultrasound medical imaging, SAR satellite images
    
    **Model Performance**:
    - This model was trained specifically on Gaussian noise (factor 0.25)
    - Best performance: Gaussian noise at similar levels to training
    - Moderate performance: Other noise types (some generalization)
    - May struggle with: Very high noise levels or mixed noise types
    
    **Tips**:
    - Try different noise levels to see model robustness
    - Noise level 0.25 matches training conditions (best results)
    - Higher noise levels are more challenging
    - Some residual noise may remain after denoising
    """)

# Sidebar additional info
with st.sidebar:
    st.markdown('---')
    st.markdown('## Current settings')
    st.info(f"""
    **Noise Type**: {noise_type}
    
    **Noise Level**: {noise_level:.2f}
    
    **Model**: Denoising AE
    
    **Training**: Gaussian noise (0.25)
    """)
