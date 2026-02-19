"""
Custom loss functions for autoencoders.

References:
- Perceptual Loss: Johnson et al. "Perceptual Losses for Real-Time Style Transfer
  and Super-Resolution" (ECCV 2016)
- SSIM: Wang et al. "Image quality assessment: from error visibility to
  structural similarity" (IEEE TIP 2004)
"""

import tensorflow as tf
from tensorflow import keras


def get_perceptual_model():
    """
    Create a VGG16 model for perceptual loss.
    Uses block3_conv3 features for good balance of detail and semantics.
    
    Returns:
        Keras Model that outputs intermediate VGG16 features
    """
    vgg = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(64, 64, 3)
    )
    vgg.trainable = False
    
    # Extract features from an intermediate layer
    layer_name = 'block3_conv3'
    perceptual_model = keras.Model(
        inputs=vgg.input,
        outputs=vgg.get_layer(layer_name).output
    )
    
    return perceptual_model


class PerceptualLoss(keras.losses.Loss):
    """
    Perceptual loss using VGG16 features.
    
    Measures the difference in high-level features rather than raw pixels,
    leading to more visually pleasing reconstructions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.perceptual_model = get_perceptual_model()
    
    def call(self, y_true, y_pred):
        y_true_features = self.perceptual_model(y_true)
        y_pred_features = self.perceptual_model(y_pred)
        return tf.reduce_mean(tf.square(y_true_features - y_pred_features))


class CombinedLoss(keras.losses.Loss):
    """
    Combined loss: MSE + SSIM + Perceptual Loss.
    
    This multi-term loss function optimizes for:
    - MSE: Pixel-level accuracy
    - SSIM: Structural similarity (perceptual quality)
    - Perceptual: High-level feature similarity
    
    Args:
        mse_weight: Weight for MSE loss (default: 0.5)
        ssim_weight: Weight for SSIM loss (default: 0.3)
        perceptual_weight: Weight for perceptual loss (default: 0.2)
    """
    
    def __init__(self, mse_weight=0.5, ssim_weight=0.3, perceptual_weight=0.2, **kwargs):
        super().__init__(**kwargs)
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_model = get_perceptual_model() if perceptual_weight > 0 else None
    
    def call(self, y_true, y_pred):
        # MSE loss
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # SSIM loss (1 - SSIM to make it a loss to minimize)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            y_true_features = self.perceptual_model(y_true)
            y_pred_features = self.perceptual_model(y_pred)
            perceptual = tf.reduce_mean(tf.square(y_true_features - y_pred_features))
        else:
            perceptual = 0.0
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * perceptual
        )
        
        return total_loss
