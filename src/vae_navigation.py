"""
VAE navigation components for Streamlit interactive exploration.

Provides modular interfaces for navigating VAE latent space.
"""

# Third-party imports
import numpy as np
import streamlit as st
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def render_latent_sliders(latent_dim, key_prefix='vae'):
    """
    Render sliders for manual latent space navigation.
    
    Args:
        latent_dim: Dimension of the latent space
        key_prefix: Prefix for slider keys (for unique identification)
    
    Returns:
        Numpy array of latent vector from slider values
    """
    st.subheader('Manual latent space navigation')
    st.write('Adjust sliders to navigate through the latent space:')
    
    # Limit displayed sliders to avoid clutter
    max_display_sliders = min(latent_dim, 10)
    
    if latent_dim > max_display_sliders:
        st.info(f'Showing {max_display_sliders} of {latent_dim} latent dimensions. '
                f'Remaining dimensions are sampled from N(0,1).')
    
    # Create sliders in columns for better layout
    cols_per_row = 2
    latent_vector = np.zeros(latent_dim)
    
    for i in range(0, max_display_sliders, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < max_display_sliders:
                with cols[j]:
                    latent_vector[idx] = st.slider(
                        f'Dim {idx + 1}',
                        min_value=-3.0,
                        max_value=3.0,
                        value=0.0,
                        step=0.1,
                        key=f'{key_prefix}_slider_{idx}'
                    )
    
    # Sample remaining dimensions from standard normal
    if latent_dim > max_display_sliders:
        latent_vector[max_display_sliders:] = np.random.randn(latent_dim - max_display_sliders)
    
    # Add preset buttons
    st.write('**Quick Presets:**')
    preset_cols = st.columns(4)
    
    with preset_cols[0]:
        if st.button('Random', key=f'{key_prefix}_random'):
            st.session_state[f'{key_prefix}_random_vector'] = np.random.randn(latent_dim)
    
    with preset_cols[1]:
        if st.button('0️⃣ Zero', key=f'{key_prefix}_zero'):
            st.session_state[f'{key_prefix}_random_vector'] = np.zeros(latent_dim)
    
    with preset_cols[2]:
        if st.button('Positive', key=f'{key_prefix}_positive'):
            st.session_state[f'{key_prefix}_random_vector'] = np.ones(latent_dim)
    
    with preset_cols[3]:
        if st.button('Negative', key=f'{key_prefix}_negative'):
            st.session_state[f'{key_prefix}_random_vector'] = -np.ones(latent_dim)
    
    # Use preset vector if available
    if f'{key_prefix}_random_vector' in st.session_state:
        latent_vector = st.session_state[f'{key_prefix}_random_vector']
        # Update only the first few dimensions with slider values
        for i in range(max_display_sliders):
            latent_vector[i] = st.session_state[f'{key_prefix}_slider_{i}']
    
    return latent_vector


def render_clickable_latent_space(encoded_images, labels, decoder, class_names=None, 
                                   reduction_method='tsne', key_prefix='vae'):
    """
    Render interactive latent space visualization with click-to-generate.
    
    Args:
        encoded_images: Encoded latent vectors (N, latent_dim)
        labels: Class labels for coloring
        decoder: VAE decoder model
        class_names: Optional list of class names
        reduction_method: 'tsne' or 'umap' for dimensionality reduction
        key_prefix: Prefix for component keys
    
    Returns:
        Selected latent vector (if clicked) or None
    """
    st.subheader('Interactive latent space map')
    
    # Reduce dimensionality for visualization
    with st.spinner(f'Computing {reduction_method.upper()} projection...'):
        if reduction_method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Use a subset for faster computation
        n_samples = min(1000, len(encoded_images))
        indices = np.random.choice(len(encoded_images), n_samples, replace=False)
        
        reduced = reducer.fit_transform(encoded_images[indices])
    
    # Create interactive scatter plot
    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=labels[indices].astype(str),
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Class'},
        title=f'Latent Space ({reduction_method.upper()} Projection)',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data={'x': ':.2f', 'y': ':.2f'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(
        height=500,
        clickmode='event+select',
        hovermode='closest'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_latent_plot')
    
    st.info('Click on a point in the latent space to generate an image from that location!')
    
    # Note: Streamlit doesn't natively support click events on plotly charts
    # This is a simplified version. For full interactivity, consider using streamlit-plotly-events
    st.write('*Note: Full click interaction requires additional setup. '
             'Use the slider interface for manual navigation.*')
    
    return None


def render_interpolation_controls(encoder, decoder, test_images, test_labels, 
                                  class_names=None, key_prefix='vae'):
    """
    Render controls for latent space interpolation between two images.
    
    Args:
        encoder: VAE encoder model
        decoder: VAE decoder model
        test_images: Test dataset images
        test_labels: Test dataset labels
        class_names: Optional list of class names
        key_prefix: Prefix for component keys
    
    Returns:
        Tuple of (interpolated_images, start_image, end_image)
    """
    st.subheader('Latent space interpolation')
    st.write('Interpolate between two images in latent space:')
    
    # Image selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Start Image**')
        
        # Option to select by class or random
        start_mode = st.radio(
            'Selection mode:',
            ['Random', 'By Class'],
            key=f'{key_prefix}_start_mode',
            horizontal=True
        )
        
        if start_mode == 'By Class' and class_names is not None:
            start_class = st.selectbox(
                'Select class:',
                range(len(class_names)),
                format_func=lambda x: class_names[x],
                key=f'{key_prefix}_start_class'
            )
            # Get random image from this class
            class_indices = np.where(test_labels == start_class)[0]
            start_idx = np.random.choice(class_indices)
        else:
            start_idx = st.number_input(
                'Image index:',
                min_value=0,
                max_value=len(test_images) - 1,
                value=0,
                key=f'{key_prefix}_start_idx'
            )
        
        start_image = test_images[start_idx:start_idx+1]
        st.image(start_image[0], caption=f'Index: {start_idx}', use_container_width=True)
    
    with col2:
        st.write('**End Image**')
        
        end_mode = st.radio(
            'Selection mode:',
            ['Random', 'By Class'],
            key=f'{key_prefix}_end_mode',
            horizontal=True
        )
        
        if end_mode == 'By Class' and class_names is not None:
            end_class = st.selectbox(
                'Select class:',
                range(len(class_names)),
                format_func=lambda x: class_names[x],
                key=f'{key_prefix}_end_class'
            )
            class_indices = np.where(test_labels == end_class)[0]
            end_idx = np.random.choice(class_indices)
        else:
            end_idx = st.number_input(
                'Image index:',
                min_value=0,
                max_value=len(test_images) - 1,
                value=100,
                key=f'{key_prefix}_end_idx'
            )
        
        end_image = test_images[end_idx:end_idx+1]
        st.image(end_image[0], caption=f'Index: {end_idx}', use_container_width=True)
    
    # Interpolation steps
    n_steps = st.slider(
        'Number of interpolation steps:',
        min_value=5,
        max_value=20,
        value=10,
        key=f'{key_prefix}_steps'
    )
    
    # Perform interpolation
    if st.button('Generate interpolation', key=f'{key_prefix}_interpolate_btn'):
        with st.spinner('Generating interpolation...'):
            # Encode both images
            start_z_mean, start_z_log_var, start_z = encoder.predict(start_image, verbose=0)
            end_z_mean, end_z_log_var, end_z = encoder.predict(end_image, verbose=0)
            
            # Use mean vectors for smooth interpolation
            start_latent = start_z_mean
            end_latent = end_z_mean
            
            # Create interpolation
            interpolated_latents = []
            for alpha in np.linspace(0, 1, n_steps):
                interpolated = (1 - alpha) * start_latent + alpha * end_latent
                interpolated_latents.append(interpolated)
            
            interpolated_latents = np.vstack(interpolated_latents)
            
            # Decode
            interpolated_images = decoder.predict(interpolated_latents, verbose=0)
            
            # Store in session state
            st.session_state[f'{key_prefix}_interpolation'] = {
                'images': interpolated_images,
                'start': start_image[0],
                'end': end_image[0]
            }
    
    # Display results if available
    if f'{key_prefix}_interpolation' in st.session_state:
        interp_data = st.session_state[f'{key_prefix}_interpolation']
        
        st.write('**Interpolation Result:**')
        
        # Display as a grid
        cols = st.columns(min(n_steps, 10))
        images_to_show = interp_data['images']
        
        # If more than 10 steps, sample evenly
        if len(images_to_show) > 10:
            indices = np.linspace(0, len(images_to_show) - 1, 10, dtype=int)
            images_to_show = images_to_show[indices]
        
        for idx, col in enumerate(cols):
            with col:
                st.image(images_to_show[idx], use_container_width=True)
        
        return interp_data['images'], interp_data['start'], interp_data['end']
    
    return None, None, None


def generate_from_latent(decoder, latent_vector):
    """
    Generate image from latent vector.
    
    Args:
        decoder: VAE decoder model
        latent_vector: Latent vector (1D array)
    
    Returns:
        Generated image
    """
    # Ensure correct shape
    if len(latent_vector.shape) == 1:
        latent_vector = latent_vector.reshape(1, -1)
    
    # Generate
    generated = decoder.predict(latent_vector, verbose=0)
    
    return generated[0]


def render_latent_arithmetic(encoder, decoder, test_images, key_prefix='vae'):
    """
    Render interface for latent space arithmetic (image1 + image2 - image3).
    
    Args:
        encoder: VAE encoder model
        decoder: VAE decoder model
        test_images: Test dataset images
        key_prefix: Prefix for component keys
    
    Returns:
        Generated image from arithmetic operation
    """
    st.subheader('Latent space arithmetic')
    st.write('Perform arithmetic operations in latent space: **A + B - C**')
    
    col1, col2, col3 = st.columns(3)
    
    # Select three images
    with col1:
        idx_a = st.number_input('Image A index:', 0, len(test_images)-1, 0, key=f'{key_prefix}_arith_a')
        st.image(test_images[idx_a], caption='A', use_container_width=True)
    
    with col2:
        idx_b = st.number_input('Image B index:', 0, len(test_images)-1, 1, key=f'{key_prefix}_arith_b')
        st.image(test_images[idx_b], caption='B', use_container_width=True)
    
    with col3:
        idx_c = st.number_input('Image C index:', 0, len(test_images)-1, 2, key=f'{key_prefix}_arith_c')
        st.image(test_images[idx_c], caption='C', use_container_width=True)
    
    if st.button('Compute A + B - C', key=f'{key_prefix}_compute_arith'):
        with st.spinner('Computing...'):
            # Encode all three images
            images = np.array([test_images[idx_a], test_images[idx_b], test_images[idx_c]])
            z_mean, _, _ = encoder.predict(images, verbose=0)
            
            # Perform arithmetic
            result_latent = z_mean[0] + z_mean[1] - z_mean[2]
            
            # Decode
            result_image = generate_from_latent(decoder, result_latent)
            
            st.write('**Result:**')
            st.image(result_image, caption='A + B - C', use_container_width=True)
            
            return result_image
    
    return None
