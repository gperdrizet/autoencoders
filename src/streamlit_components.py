"""
Reusable Streamlit UI components for the autoencoder demo app.
"""

# Standard library imports
import io

# Third-party imports
import numpy as np
import streamlit as st
from PIL import Image


def render_header(title, description):
    """
    Render a consistent page header.
    
    Args:
        title: Page title
        description: Page description
    """
    st.title(title)
    st.markdown(description)
    st.markdown('---')


def render_model_info_sidebar(model, model_name='Model'):
    """
    Render model information in sidebar.
    
    Args:
        model: Keras model
        model_name: Display name for the model
    """
    with st.sidebar:
        st.subheader(f'{model_name} info')
        
        total_params = model.count_params()
        
        st.metric("Total Parameters", f"{total_params:,}")
        st.metric('Layers', len(model.layers))
        
        if hasattr(model, 'input_shape'):
            st.text(f'Input: {model.input_shape}')
        if hasattr(model, 'output_shape'):
            st.text(f'Output: {model.output_shape}')
        
        with st.expander('View Architecture'):
            # Create a string representation of model summary
            string_buffer = io.StringIO()
            model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
            st.text(string_buffer.getvalue())


def render_image_uploader(key='image_upload'):
    """
    Render image upload widget with preview.
    
    Args:
        key: Unique key for the uploader
    
    Returns:
        Uploaded file object or None
    """
    st.subheader('Upload your own image')
    
    uploaded_file = st.file_uploader(
        'Choose an image file (will be resized to 32x32)',
        type=['png', 'jpg', 'jpeg'],
        key=key
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Original Upload', use_container_width=True)
        
        with col2:
            # Show what it will look like at 32x32
            preview = image.resize((32, 32), Image.LANCZOS)
            preview_display = preview.resize((128, 128), Image.NEAREST)  # Scale up for visibility
            st.image(preview_display, caption='32x32 Preview (scaled)', use_container_width=True)
    
    return uploaded_file


def render_sample_selector(x_data, y_data, class_names, key='sample'):
    """
    Render sample image selector with class filter.
    
    Args:
        x_data: Image dataset
        y_data: Labels
        class_names: List of class names
        key: Unique key prefix
    
    Returns:
        Tuple of (selected_image, selected_label, selected_index)
    """
    st.subheader('Select from dataset')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Class filter
        selected_class = st.selectbox(
            'Filter by class:',
            ['All'] + class_names,
            key=f'{key}_class'
        )
        
        # Get indices for selected class
        if selected_class == 'All':
            valid_indices = np.arange(len(x_data))
        else:
            class_idx = class_names.index(selected_class)
            valid_indices = np.where(y_data == class_idx)[0]
        
        # Random sample button
        if st.button('Random sample', key=f'{key}_random'):
            st.session_state[f'{key}_idx'] = np.random.choice(valid_indices)
    
    with col2:
        # Index selector
        if f"{key}_idx" not in st.session_state:
            st.session_state[f"{key}_idx"] = valid_indices[0]
        
        selected_idx = st.number_input(
            'Image index:',
            min_value=int(valid_indices.min()),
            max_value=int(valid_indices.max()),
            value=int(st.session_state[f'{key}_idx']),
            key=f'{key}_number'
        )
    
    selected_image = x_data[selected_idx]
    selected_label = y_data[selected_idx]
    
    # Display selected image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(selected_image, caption=f"{class_names[selected_label]} (#{selected_idx})", 
                use_container_width=True)
    
    return selected_image, selected_label, selected_idx


def create_download_button(image, filename='image.png', button_text='Download Image'):
    """
    Create a download button for an image.
    
    Args:
        image: Numpy array (normalized [0,1] or uint8)
        filename: Default filename for download
        button_text: Button label
    """
    # Convert to PIL Image
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Convert to bytes
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    byte_data = buf.getvalue()
    
    st.download_button(
        label=button_text,
        data=byte_data,
        file_name=filename,
        mime='image/png'
    )


def render_metrics_display(metrics_dict, columns=3):
    """
    Display metrics in a clean grid layout.
    
    Args:
        metrics_dict: Dictionary of metric_name: value
        columns: Number of columns for layout
    """
    cols = st.columns(columns)
    
    for idx, (metric_name, value) in enumerate(metrics_dict.items()):
        with cols[idx % columns]:
            # Format value based on type
            if isinstance(value, float):
                if value < 0.01:
                    formatted_value = f'{value:.2e}'
                else:
                    formatted_value = f'{value:.4f}'
            else:
                formatted_value = str(value)
            
            st.metric(
                label=metric_name.replace('_', ' ').title(),
                value=formatted_value
            )


def render_explanation_expander(title, content):
    """
    Render an expandable explanation section.
    
    Args:
        title: Expander title
        content: Markdown content
    """
    with st.expander(title):
        st.markdown(content)


def render_sidebar_navigation():
    """
    Render navigation info in sidebar.
    """
    with st.sidebar:
        st.markdown('## Navigation')
        st.markdown("""
        **Demos available:**
        - **Compression** - Image compression and reconstruction
        - **Anomaly detection** - Detect unusual images
        - **VAE generation** - Generate new images
        
        Use the sidebar to navigate between demos.
        """)


def show_loading_message(message='Processing...'):
    """
    Context manager for showing loading messages.
    
    Args:
        message: Loading message to display
    
    Returns:
        Context manager (use with 'with' statement)
    """
    return st.spinner(message)


def render_code_snippet(code, language='python'):
    """
    Render a code snippet with syntax highlighting.
    
    Args:
        code: Code string
        language: Programming language for syntax highlighting
    """
    st.code(code, language=language)


def render_warning(message, icon=''):
    """
    Display a formatted warning message.
    
    Args:
        message: Warning message
        icon: Icon to display
    """
    if icon:
        st.warning(f'{icon} {message}')
    else:
        st.warning(message)


def render_success(message, icon=''):
    """
    Display a formatted success message.
    
    Args:
        message: Success message
        icon: Icon to display
    """
    if icon:
        st.success(f'{icon} {message}')
    else:
        st.success(message)


def render_info(message, icon=''):
    """
    Display a formatted info message.
    
    Args:
        message: Info message
        icon: Icon to display
    """
    if icon:
        st.info(f'{icon} {message}')
    else:
        st.info(message)
