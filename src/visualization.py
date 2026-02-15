"""
Visualization utilities supporting both Matplotlib (notebooks) and Plotly (Streamlit).
"""

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_image_grid(images, titles=None, rows=2, cols=5, figsize=(15, 6), cmap=None):
    """
    Plot a grid of images using Matplotlib (for notebooks).
    
    Args:
        images: Array of images
        titles: Optional list of titles
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        cmap: Colormap (None for RGB)
    
    Returns:
        Matplotlib figure
    """
    n_images = min(len(images), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap=cmap)
        axes[i].axis('off')
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_reconstruction_comparison(original, reconstructed, n_samples=5):
    """
    Plot original vs reconstructed images side by side (Matplotlib).
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        n_samples: Number of samples to display
    
    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, len(original))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontweight='bold')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_difference_heatmap(original, reconstructed):
    """
    Plot difference heatmap between original and reconstructed (Matplotlib).
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
    
    Returns:
        Matplotlib figure
    """
    # Calculate absolute difference
    diff = np.abs(original - reconstructed)
    # Average across color channels
    diff_gray = np.mean(diff, axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    im = axes[2].imshow(diff_gray, cmap='hot')
    axes[2].set_title('Difference (Heatmap)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plot training history (Matplotlib for notebooks).
    
    Args:
        history: Keras training history object
    
    Returns:
        Matplotlib figure
    """
    metrics = list(history.history.keys())
    n_metrics = len([m for m in metrics if not m.startswith('val_')])
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate([m for m in metrics if not m.startswith('val_')]):
        axes[idx].plot(history.history[metric], label=f'Train {metric}')
        if f'val_{metric}' in metrics:
            axes[idx].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latent_space_2d(encoded_images, labels, class_names=None):
    """
    Plot 2D latent space visualization (Matplotlib).
    
    Args:
        encoded_images: Encoded latent vectors (N, 2)
        labels: Class labels
        class_names: Optional list of class names
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        encoded_images[:, 0],
        encoded_images[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    
    ax.set_xlabel('Latent dimension 1')
    ax.set_ylabel('Latent dimension 2')
    ax.set_title('2D latent space visualization')
    
    if class_names is not None:
        plt.colorbar(scatter, ax=ax, ticks=range(len(class_names)), label='Class')
    else:
        plt.colorbar(scatter, ax=ax, label='Class')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_plotly_comparison(original, reconstructed, title='Comparison'):
    """
    Create Plotly side-by-side comparison (for Streamlit).
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        title: Figure title
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original', 'Reconstructed'),
        horizontal_spacing=0.05
    )
    
    # Original
    fig.add_trace(
        go.Image(z=(original * 255).astype(np.uint8)),
        row=1, col=1
    )
    
    # Reconstructed
    fig.add_trace(
        go.Image(z=(reconstructed * 255).astype(np.uint8)),
        row=1, col=2
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=400
    )
    
    return fig


def create_plotly_heatmap(original, reconstructed):
    """
    Create Plotly difference heatmap (for Streamlit).
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
    
    Returns:
        Plotly figure
    """
    diff = np.abs(original - reconstructed)
    diff_gray = np.mean(diff, axis=2)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Original', 'Reconstructed', 'Difference'),
        horizontal_spacing=0.05
    )
    
    # Original
    fig.add_trace(
        go.Image(z=(original * 255).astype(np.uint8)),
        row=1, col=1
    )
    
    # Reconstructed
    fig.add_trace(
        go.Image(z=(reconstructed * 255).astype(np.uint8)),
        row=1, col=2
    )
    
    # Difference heatmap
    fig.add_trace(
        go.Heatmap(
            z=diff_gray,
            colorscale='Hot',
            showscale=True
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        title_text='Reconstruction analysis',
        height=400
    )
    
    return fig


def create_plotly_roc_curve(fpr, tpr, auc_score, threshold=None):
    """
    Create Plotly ROC curve (for Streamlit).
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC score
        threshold: Optional threshold to mark on curve
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC curve - Anomaly detection',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        hovermode='closest'
    )
    
    return fig


def create_plotly_histogram(data, title='Distribution', bins=50):
    """
    Create Plotly histogram (for Streamlit).
    
    Args:
        data: Data to plot
        title: Plot title
        bins: Number of bins
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name='Distribution'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Frequency',
        showlegend=False,
        height=400
    )
    
    return fig


def create_plotly_latent_space(encoded_images, labels, class_names=None):
    """
    Create interactive Plotly latent space visualization (for Streamlit).
    
    Args:
        encoded_images: Encoded latent vectors (N, 2 or 3)
        labels: Class labels
        class_names: Optional list of class names
    
    Returns:
        Plotly figure
    """
    if encoded_images.shape[1] == 2:
        # 2D scatter
        fig = px.scatter(
            x=encoded_images[:, 0],
            y=encoded_images[:, 1],
            color=labels,
            labels={'x': 'Latent dim 1', 'y': 'Latent dim 2', 'color': 'Class'},
            title='2D latent space',
            color_continuous_scale='viridis'
        )
    elif encoded_images.shape[1] >= 3:
        # 3D scatter
        fig = px.scatter_3d(
            x=encoded_images[:, 0],
            y=encoded_images[:, 1],
            z=encoded_images[:, 2],
            color=labels,
            labels={'x': 'Latent dim 1', 'y': 'Latent dim 2', 'z': 'Latent dim 3', 'color': 'Class'},
            title='3D latent space',
            color_continuous_scale='viridis'
        )
    else:
        raise ValueError('Encoded images must have at least 2 dimensions')
    
    fig.update_layout(height=600)
    
    return fig


def plot_interpolation(images, n_steps):
    """
    Plot interpolation sequence (Matplotlib for notebooks).
    
    Args:
        images: Interpolated images
        n_steps: Number of interpolation steps
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
    
    for i in range(n_steps):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title('Start')
        elif i == n_steps - 1:
            axes[i].set_title('End')
    
    plt.tight_layout()
    return fig
