"""
Anomaly Detection Demo - Streamlit Page

Interactive demonstration of anomaly detection using autoencoders.
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
from src.data_utils import FLOWER_CLASSES, COCO_CLASSES, load_flowers, load_coco
from src.metrics import (
    calculate_reconstruction_error,
    compute_anomaly_threshold,
    compute_roc_metrics,
    find_optimal_threshold
)
from src.model_utils import load_model
from src.visualization import create_plotly_heatmap, create_plotly_histogram, create_plotly_roc_curve
from src.streamlit_components import (
    render_header,
    render_model_info_sidebar,
    render_explanation_expander,
    render_metrics_display,
    show_loading_message
)

# Page config
st.set_page_config(
    page_title='Anomaly Detection',
    page_icon='⬛',
    layout='wide'
)

# Header
render_header(
    'Anomaly detection with autoencoders',
    'Identify unusual patterns using reconstruction error'
)

# Explanation
render_explanation_expander(
    'How does anomaly detection work?',
    """
    Autoencoders can detect anomalies using a simple but powerful idea:
    
    **Training**:
    1. Train the autoencoder ONLY on "normal" data
    2. The model learns to reconstruct normal patterns very well
    
    **Detection**:
    1. Pass test data (normal + anomalies) through the autoencoder
    2. Calculate reconstruction error for each image
    3. High error = anomaly, Low error = normal
    
    **Why it works**:
    - The model hasn't seen anomalous patterns during training
    - It struggles to reconstruct unfamiliar (anomalous) data
    - Reconstruction error serves as an anomaly score
    
    **Key Concepts**:
    - **Threshold**: Error value above which we classify as anomaly
    - **ROC Curve**: Shows trade-off between true positives and false positives
    - **AUC**: Area under ROC curve (higher is better, 1.0 is perfect)
    
    **In this demo**:
    - Normal data: All 5 flower classes (dandelion, daisy, tulips, sunflowers, roses)
    - Anomalous data: Random COCO images (people, vehicles, animals, objects, etc.)
    - The model was trained only on flowers
    - It should flag non-flower images as anomalies
    """
)

st.markdown('---')

# Load model
@st.cache_resource
def load_anomaly_model():
    from src.huggingface_utils import download_model
    
    model_name = 'anomaly_ae.keras'
    try:
        model_path = download_model(model_name, models_dir='models')
        return load_model(model_path)
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.info('Make sure the model is uploaded to Hugging Face or train it locally.')
        st.stop()

with show_loading_message('Loading anomaly detection model...'):
    model = load_anomaly_model()

# Show model info
render_model_info_sidebar(model, 'Anomaly Detection AE')

# Load dataset
@st.cache_data
def load_anomaly_data():
    # Load flowers as normal data
    (x_train_flowers, y_train_flowers), (x_test_flowers, y_test_flowers) = load_flowers(normalize=True)
    
    # Load COCO subset as anomalous data (use 500 random samples for testing)
    (x_train_coco, y_train_coco), (x_test_coco, y_test_coco) = load_coco(subset_percent=10, normalize=True)
    
    # Sample 500 random COCO images for anomaly testing
    n_anomaly_samples = min(500, len(x_test_coco))
    anomaly_indices = np.random.RandomState(42).choice(len(x_test_coco), n_anomaly_samples, replace=False)
    
    return {
        'x_train_normal': x_train_flowers,
        'y_train_normal': y_train_flowers,
        'x_test_normal': x_test_flowers,
        'y_test_normal': y_test_flowers,
        'x_test_anomaly': x_test_coco[anomaly_indices],
        'y_test_anomaly': y_test_coco[anomaly_indices],
    }

data = load_anomaly_data()

# Compute reconstruction errors
@st.cache_data
def compute_errors(_model):
    """Compute reconstruction errors for normal and anomalous data."""
    errors_normal = calculate_reconstruction_error(model, data['x_test_normal'])
    errors_anomaly = calculate_reconstruction_error(model, data['x_test_anomaly'])
    return errors_normal, errors_anomaly

with show_loading_message('Computing reconstruction errors...'):
    errors_normal, errors_anomaly = compute_errors(model)

# Compute ROC metrics
y_true = np.concatenate([
    np.zeros(len(errors_normal)),
    np.ones(len(errors_anomaly))
])
scores = np.concatenate([errors_normal, errors_anomaly])
roc_metrics = compute_roc_metrics(y_true, scores)
optimal_threshold = find_optimal_threshold(y_true, scores)

# Sidebar configuration
st.sidebar.markdown('## Detection settings')

# Threshold selection method
threshold_method = st.sidebar.radio(
    'Threshold Selection:',
    ['Optimal (ROC-based)', 'Percentile-based', 'Manual'],
    help='Choose how to set the anomaly detection threshold'
)

if threshold_method == 'Optimal (ROC-based)':
    threshold = optimal_threshold
    st.sidebar.success(f'Optimal Threshold: {threshold:.6f}')
elif threshold_method == 'Percentile-based':
    percentile = st.sidebar.slider('Percentile:', 90, 99, 95, 1)
    threshold = compute_anomaly_threshold(errors_normal, percentile=percentile)
    st.sidebar.info(f'{percentile}th Percentile: {threshold:.6f}')
else:
    threshold = st.sidebar.slider(
        'Manual Threshold:',
        float(scores.min()),
        float(scores.max()),
        float(optimal_threshold),
        step=0.0001,
        format='%.6f'
    )

# Calculate predictions with current threshold
y_pred = (scores > threshold).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# Main content
st.markdown('## Overall performance')

# Display metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('ROC-AUC', f'{roc_metrics["auc"]:.4f}')
col2.metric('Accuracy', f'{accuracy:.4f}')
col3.metric('Precision', f'{precision:.4f}')
col4.metric('Recall', f'{recall:.4f}')
col5.metric('F1-Score', f'{f1:.4f}')

st.markdown('---')

# ROC Curve and Distribution
st.markdown('## Analysis')

tab1, tab2, tab3 = st.tabs(['ROC Curve', 'Error Distribution', 'Confusion Matrix'])

with tab1:
    st.markdown('### ROC Curve')
    st.markdown('Shows the trade-off between True Positive Rate and False Positive Rate')
    
    fig_roc = create_plotly_roc_curve(
        roc_metrics['fpr'],
        roc_metrics['tpr'],
        roc_metrics['auc'],
        threshold=threshold
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.info(f'AUC = {roc_metrics['auc']:.4f}: ' + 
            ('Excellent performance! ' if roc_metrics['auc'] > 0.95 else
             'Very good performance! ' if roc_metrics['auc'] > 0.9 else
             'Good performance. ' if roc_metrics['auc'] > 0.8 else
             'Moderate performance. ') +
            'The model can effectively distinguish between normal and anomalous images.')

with tab2:
    st.markdown('### Reconstruction Error Distribution')
    st.markdown('Compare error distributions for normal vs anomalous data')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_normal = create_plotly_histogram(
            errors_normal,
            title='Normal Data - Reconstruction Errors',
            bins=50
        )
        st.plotly_chart(fig_normal, use_container_width=True)
        
        st.metric('Mean Error (Normal)', f'{np.mean(errors_normal):.6f}')
        st.metric('Std Error (Normal)', f'{np.std(errors_normal):.6f}')
    
    with col2:
        fig_anomaly = create_plotly_histogram(
            errors_anomaly,
            title='Anomalous Data - Reconstruction Errors',
            bins=50
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        st.metric('Mean Error (Anomaly)', f'{np.mean(errors_anomaly):.6f}')
        st.metric('Std Error (Anomaly)', f'{np.std(errors_anomaly):.6f}')
    
    # Show threshold line
    st.markdown(f'**Current Threshold**: {threshold:.6f}')
    separation = (np.mean(errors_anomaly) - np.mean(errors_normal)) / np.mean(errors_normal) * 100
    st.success(f'Anomalous errors are {separation:.1f}% higher than normal errors on average')

with tab3:
    st.markdown('### Confusion Matrix')
    
    # Display confusion matrix
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('#### Matrix')
        tn, fp, fn, tp = cm.ravel()
        
        # Create a formatted table
        import pandas as pd
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            columns=['Predicted Normal', 'Predicted Anomaly'],
            index=['Actual Normal', 'Actual Anomaly']
        )
        st.dataframe(cm_df, use_container_width=True)
    
    with col2:
        st.markdown('#### Interpretation')
        st.markdown(f"""
        - **True Negatives (TN)**: {tn} - Correctly identified normal images
        - **False Positives (FP)**: {fp} - Normal images incorrectly flagged as anomalies
        - **False Negatives (FN)**: {fn} - Anomalies that were missed
        - **True Positives (TP)**: {tp} - Correctly detected anomalies
        
        **Performance:**
        - **Precision** ({precision:.3f}): Of all flagged anomalies, {precision*100:.1f}% were actual anomalies
        - **Recall** ({recall:.3f}): Of all actual anomalies, {recall*100:.1f}% were detected
        """)

st.markdown('---')

# Sample Analysis
st.markdown('## Sample analysis')

analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
    'Top Anomalies', 'False Negatives', 'False Positives'
])

with analysis_tab1:
    st.markdown('### Highest Anomaly Scores (Correctly Detected)')
    st.markdown('These anomalous images have the highest reconstruction errors')
    
    # Get top 10 anomalies
    top_indices = np.argsort(errors_anomaly)[-10:][::-1]
    
    cols = st.columns(5)
    for i, idx in enumerate(top_indices[:5]):
        with cols[i]:
            st.image(data['x_test_anomaly'][idx], use_container_width=True)
            st.caption(f'Error: {errors_anomaly[idx]:.4f}')
    
    cols = st.columns(5)
    for i, idx in enumerate(top_indices[5:10]):
        with cols[i]:
            st.image(data['x_test_anomaly'][idx], use_container_width=True)
            st.caption(f'Error: {errors_anomaly[idx]:.4f}')

with analysis_tab2:
    st.markdown('### False Negatives (Missed Anomalies)')
    st.markdown('Anomalous images with low reconstruction errors (below threshold)')
    
    # Get anomalies below threshold
    missed_mask = errors_anomaly < threshold
    if missed_mask.sum() > 0:
        missed_indices = np.where(missed_mask)[0]
        missed_errors = errors_anomaly[missed_mask]
        sorted_missed = missed_indices[np.argsort(missed_errors)[:10]]
        
        cols = st.columns(5)
        for i, idx in enumerate(sorted_missed[:5]):
            if i < len(sorted_missed):
                with cols[i]:
                    st.image(data['x_test_anomaly'][idx], use_container_width=True)
                    st.caption(f'Error: {errors_anomaly[idx]:.4f}')
        
        cols = st.columns(5)
        for i, idx in enumerate(sorted_missed[5:10]):
            if i < len(sorted_missed[5:]):
                with cols[i]:
                    st.image(data['x_test_anomaly'][idx], use_container_width=True)
                    st.caption(f'Error: {errors_anomaly[idx]:.4f}')
        
        st.warning(f'{missed_mask.sum()} anomalies ({missed_mask.sum()/len(errors_anomaly)*100:.1f}%) were not detected')
    else:
        st.success('All anomalies were detected with the current threshold!')

with analysis_tab3:
    st.markdown('### False Positives (Normal Images Flagged)')
    st.markdown('Normal images with high reconstruction errors (above threshold)')
    
    # Get normal images above threshold
    flagged_mask = errors_normal > threshold
    if flagged_mask.sum() > 0:
        flagged_indices = np.where(flagged_mask)[0]
        flagged_errors = errors_normal[flagged_mask]
        sorted_flagged = flagged_indices[np.argsort(flagged_errors)[-10:][::-1]]
        
        cols = st.columns(5)
        for i, idx in enumerate(sorted_flagged[:5]):
            if i < len(sorted_flagged):
                with cols[i]:
                    st.image(data['x_test_normal'][idx], use_container_width=True)
                    st.caption(f'Error: {errors_normal[idx]:.4f}')
        
        cols = st.columns(5)
        for i, idx in enumerate(sorted_flagged[5:10]):
            if i < len(sorted_flagged[5:]):
                with cols[i]:
                    st.image(data['x_test_normal'][idx], use_container_width=True)
                    st.caption(f'Error: {errors_normal[idx]:.4f}')
        
        st.warning(f'{flagged_mask.sum()} normal images ({flagged_mask.sum()/len(errors_normal)*100:.1f}%) were incorrectly flagged')
    else:
        st.success('No normal images were incorrectly flagged!')

# Detailed image analysis
st.markdown('---')
st.markdown('## Detailed image analysis')

col1, col2 = st.columns(2)

with col1:
    st.markdown('### Analyze Normal Image')
    normal_idx = st.number_input(
        'Select normal image index:',
        0, len(data['x_test_normal']) - 1, 0,
        key='normal_idx'
    )
    
    normal_img = data['x_test_normal'][normal_idx:normal_idx+1]
    normal_recon = model.predict(normal_img, verbose=0)
    normal_error = errors_normal[normal_idx]
    
    st.image(normal_img[0], caption='Normal Image', use_container_width=True)
    
    is_flagged = normal_error > threshold
    if is_flagged:
        st.error(f'Flagged as anomaly (Error: {normal_error:.6f})')
    else:
        st.success(f'Classified as normal (Error: {normal_error:.6f})')
    
    st.metric('Reconstruction Error', f'{normal_error:.6f}')
    
    # Show reconstruction
    fig = create_plotly_heatmap(normal_img[0], normal_recon[0])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('### Analyze Anomalous Image')
    anomaly_idx = st.number_input(
        'Select anomaly image index:',
        0, len(data['x_test_anomaly']) - 1, 0,
        key='anomaly_idx'
    )
    
    anomaly_img = data['x_test_anomaly'][anomaly_idx:anomaly_idx+1]
    anomaly_recon = model.predict(anomaly_img, verbose=0)
    anomaly_error = errors_anomaly[anomaly_idx]
    
    st.image(anomaly_img[0], caption='Anomalous Image', use_container_width=True)
    
    is_detected = anomaly_error > threshold
    if is_detected:
        st.success(f'Detected as anomaly (Error: {anomaly_error:.6f})')
    else:
        st.error(f'Missed (Error: {anomaly_error:.6f})')
    
    st.metric('Reconstruction Error', f'{anomaly_error:.6f}')
    
    # Show reconstruction
    fig = create_plotly_heatmap(anomaly_img[0], anomaly_recon[0])
    st.plotly_chart(fig, use_container_width=True)

# Educational content
st.markdown('---')
with st.expander('Understanding anomaly detection'):
    st.markdown("""
    **Key Insights:**
    
    1. **Threshold Matters**: The choice of threshold determines the trade-off between:
        - **High threshold**: Fewer false alarms, but may miss some anomalies
        - **Low threshold**: Catch more anomalies, but more false alarms
    
    2. **ROC-AUC Score**: Measures overall detection capability
        - 1.0 = Perfect detection
        - 0.9-1.0 = Excellent
        - 0.8-0.9 = Very good
        - 0.7-0.8 = Good
        - < 0.7 = May need improvement
    
    3. **Precision vs Recall**:
        - **High Precision**: When flagged, it's likely a real anomaly (fewer false alarms)
        - **High Recall**: Most real anomalies are caught (fewer misses)
        - Usually can't maximize both - choose based on use case
    
    4. **Real-World Applications**:
        - **Manufacturing**: High recall (don't miss defects)
        - **Security**: Balanced (neither miss threats nor overwhelm with false alarms)
        - **Medical**: High recall (don't miss diseases)
    
    **Limitations:**
    - Requires clean "normal" training data
    - May struggle with subtle anomalies
    - Threshold selection can be challenging
    - New types of anomalies may not be detected
    """)

# Sidebar stats
with st.sidebar:
    st.markdown('---')
    st.markdown('## Dataset stats')
    st.info(f"""
    **Normal Test Samples**: {len(errors_normal)}
    
    **Anomalous Test Samples**: {len(errors_anomaly)}
    
    **Current Threshold**: {threshold:.6f}
    
    **Detection Rate**: {recall*100:.1f}%
    
    **False Alarm Rate**: {fp/(fp+tn)*100:.1f}%
    """)
