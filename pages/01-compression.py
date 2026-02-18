"""
Image Compression Demo - Streamlit Page

Interactive demonstration of image compression using autoencoders.
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
from src.data_utils import FLOWER_CLASSES, load_flowers, preprocess_image
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
    page_title='Compression Demo',
    page_icon='⬛',
    layout='wide'
)

# Header
render_header(
    'Image compression with autoencoders',
    'Explore how autoencoders can compress images while maintaining quality'
)

# Explanation
render_explanation_expander(
    'How does compression work?',
    """
    An autoencoder compresses images by:
    
    1. **Encoding**: The encoder network reduces the 64x64x3 (12,288) image to a smaller latent vector
    2. **Bottleneck**: Information is forced through this compressed representation
    3. **Decoding**: The decoder reconstructs the image from the latent vector
    
    **Compression Ratio**: Original size / Latent size
    - Latent 32: ~384x compression (12,288 / 32)
    - Latent 64: ~192x compression
    - Latent 128: ~96x compression
    - Latent 256: ~48x compression
    
    **Trade-off**: Higher compression = more information loss = lower quality
    
    **Quality Metrics**:
    - **MSE** (Mean Squared Error): Lower is better
    - **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (>30 dB is good)
    - **SSIM** (Structural Similarity): Higher is better (0-1 scale, >0.9 is excellent)
    """
)

st.markdown('---')

# Model selection
st.sidebar.markdown('## Configuration')

latent_dims = [32, 64, 128, 256]
selected_latent = st.sidebar.selectbox(
    'Select Compression Level:',
    latent_dims,
    index=2,  # Default to 128
    format_func=lambda x: f'Latent {x} ({12288/x:.1f}x compression)'
)

# Load model
@st.cache_resource
def load_compression_model(latent_dim):
    from src.huggingface_utils import download_model
    
    model_name = f'compression_ae_latent{latent_dim}.keras'
    try:
        model_path = download_model(model_name, models_dir='models')
        return load_model(model_path)
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.info('Make sure the models are uploaded to Hugging Face or train them locally.')
        st.stop()

with show_loading_message(f'Loading compression model (latent {selected_latent})...'):
    model = load_compression_model(selected_latent)

# Show model info in sidebar
render_model_info_sidebar(model, f'Compression AE (Latent {selected_latent})')

# Load dataset
@st.cache_data
def load_dataset():
    (x_train, y_train), (x_test, y_test) = load_flowers(normalize=True)
    return x_test, y_test

x_test, y_test = load_dataset()

# Main content
st.markdown('## Select an image')

# Tabs for different input methods
tab1, tab2 = st.tabs(['From dataset', 'Upload your own'])

with tab1:
    selected_image, selected_label, selected_idx = render_sample_selector(
        x_test, y_test, FLOWER_CLASSES, key='compression'
    )
    input_image = selected_image

with tab2:
    uploaded_file = render_image_uploader(key='compression_upload')
    if uploaded_file is not None:
        from src.data_utils import upload_and_preprocess
        input_image = upload_and_preprocess(uploaded_file)[0]
    else:
        st.info('Upload an image to see compression results')
        input_image = None

# Process and display results
if input_image is not None:
    st.markdown('---')
    st.markdown('## Compression results')
    
    # Prepare for prediction
    input_batch = np.expand_dims(input_image, axis=0) if len(input_image.shape) == 3 else input_image.reshape(1, 32, 32, 3)
    
    # Reconstruct
    with show_loading_message('Compressing and reconstructing...'):
        reconstructed = model.predict(input_batch, verbose=0)[0]
    
    # Calculate metrics
    metrics = compute_metrics_summary(
        input_batch,
        np.expand_dims(reconstructed, axis=0),
        selected_latent
    )
    
    # Display metrics
    st.markdown('### Quality metrics')
    render_metrics_display({
        'Compression Ratio': f'{metrics["compression_ratio"]:.1f}x',
        'MSE': f'{metrics["mse"]:.6f}',
        'PSNR': f'{metrics["psnr"]:.2f} dB',
        'SSIM': f'{metrics["ssim"]:.4f}'
    }, columns=4)
    
    st.markdown('---')
    
    # Visualization options
    viz_mode = st.radio(
        'Visualization Mode:',
        ['Side-by-Side', 'Difference Heatmap'],
        horizontal=True
    )
    
    if viz_mode == 'Side-by-Side':
        fig = create_plotly_comparison(
            input_image if len(input_image.shape) == 3 else input_batch[0],
            reconstructed,
            title=f'Compression with Latent Dimension {selected_latent}'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_plotly_heatmap(
            input_image if len(input_image.shape) == 3 else input_batch[0],
            reconstructed
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download buttons
    st.markdown('### Download results')
    col1, col2 = st.columns(2)
    
    with col1:
        create_download_button(
            input_image if len(input_image.shape) == 3 else input_batch[0],
            filename=f'original_{selected_idx}.png',
            button_text='Download Original'
        )
    
    with col2:
        create_download_button(
            reconstructed,
            filename=f'compressed_latent{selected_latent}_{selected_idx}.png',
            button_text='Download Reconstructed'
        )
    
    # Additional analysis
    with st.expander('Detailed analysis'):
        st.markdown(f"""
        **Compression Details:**
        - Original size: 12,288 values (64 x 64 x 3)
        - Latent size: {selected_latent} values
        - Size reduction: {100 * (1 - selected_latent/12288):.1f}%
        
        **Quality Assessment:**
        - MSE of {metrics['mse']:.6f} indicates {'low' if metrics['mse'] < 0.01 else 'moderate' if metrics['mse'] < 0.02 else 'high'} reconstruction error
        - PSNR of {metrics['psnr']:.2f} dB is {'excellent' if metrics['psnr'] > 30 else 'good' if metrics['psnr'] > 25 else 'moderate'}
        - SSIM of {metrics['ssim']:.4f} shows {'excellent' if metrics['ssim'] > 0.9 else 'good' if metrics['ssim'] > 0.8 else 'moderate'} structural similarity
        
        **Trade-offs:**
        - Lower latent dimensions = higher compression but lower quality
        - Higher latent dimensions = lower compression but better quality
        - For this image, latent {selected_latent} provides a {['very aggressive', 'aggressive', 'balanced', 'conservative'][latent_dims.index(selected_latent)]} compression
        """)

# Comparison section
st.markdown('---')
st.markdown('## Compare compression levels')

if st.button('Generate Comparison Across All Latent Dimensions'):
    # Use currently selected image
    if input_image is not None:
        input_batch = np.expand_dims(input_image, axis=0) if len(input_image.shape) == 3 else input_image.reshape(1, 32, 32, 3)
        
        cols = st.columns(len(latent_dims) + 1)
        
        # Original
        with cols[0]:
            st.image(input_image if len(input_image.shape) == 3 else input_batch[0], 
                    caption='Original', use_container_width=True)
            st.metric('Latent Dim', 'N/A')
            st.metric('PSNR', '∞ dB')
        
        # Each latent dimension
        for idx, latent_dim in enumerate(latent_dims, start=1):
            with show_loading_message(f'Processing latent {latent_dim}...'):
                model_comp = load_compression_model(latent_dim)
                reconstructed_comp = model_comp.predict(input_batch, verbose=0)[0]
                metrics_comp = compute_metrics_summary(input_batch, np.expand_dims(reconstructed_comp, axis=0), latent_dim)
            
            with cols[idx]:
                st.image(reconstructed_comp, 
                        caption=f'Latent {latent_dim}', use_container_width=True)
                st.metric('Compression', f'{metrics_comp["compression_ratio"]:.1f}x')
                st.metric('PSNR', f'{metrics_comp["psnr"]:.1f} dB')
                st.metric('SSIM', f'{metrics_comp["ssim"]:.3f}')
    else:
        st.warning('Please select or upload an image first!')

# Educational content
st.markdown('---')
with st.expander('Tips for best results'):
    st.markdown("""
    **When to use different compression levels:**
    
    - **Latent 32** (Highest compression):
        - Maximum size reduction
        - Acceptable for thumbnails or previews
        - Best for simple images with low detail
    
    - **Latent 64**:
        - High compression with moderate quality
        - Good balance for web thumbnails
        - Works well for most images
    
    - **Latent 128** (Recommended):
        - Balanced compression and quality
        - Excellent for general use
        - Good quality retention for most images
    
    - **Latent 256** (Lowest compression):
        - Maximum quality preservation
        - Lower compression ratio
        - Best for detailed or important images
    
    **Understanding the Metrics:**
    
    - **PSNR > 30 dB**: Excellent quality, minimal visible artifacts
    - **PSNR 25-30 dB**: Good quality, some artifacts may be visible
    - **PSNR < 25 dB**: Noticeable quality degradation
    
    - **SSIM > 0.9**: Structural similarity is very high
    - **SSIM 0.8-0.9**: Good structural preservation
    - **SSIM < 0.8**: Noticeable structural differences
    """)

# Sidebar additional info
with st.sidebar:
    st.markdown('---')
    st.markdown('## Current settings')
    st.info(f"""
    **Latent Dimension**: {selected_latent}
    
    **Compression Ratio**: {12288/selected_latent:.1f}x
    
    **Size Reduction**: {100 * (1 - selected_latent/12288):.1f}%
    """)
