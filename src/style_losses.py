"""Loss functions for neural style transfer."""

import tensorflow as tf
from tensorflow import keras


def gram_matrix(features):
    """
    Compute Gram matrix for style representation.
    
    The Gram matrix captures style by computing correlations between
    different feature channels.
    
    Args:
        features: Feature tensor of shape (batch, height, width, channels)
    
    Returns:
        Gram matrix of shape (batch, channels, channels)
    """
    # Reshape: (batch, h, w, c) -> (batch, h*w, c)
    shape = tf.shape(features)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    
    features = tf.reshape(features, [batch_size, height * width, channels])
    
    # Compute Gram matrix: G = F^T * F
    gram = tf.matmul(features, features, transpose_a=True)
    
    # Normalize by number of elements
    gram = gram / tf.cast(height * width * channels, tf.float32)
    
    return gram


def content_loss(content_features, generated_features):
    """
    Content loss measures how much the generated image preserves content.
    
    Args:
        content_features: Features from content image
        generated_features: Features from generated image
    
    Returns:
        Content loss value
    """
    return tf.reduce_mean(tf.square(content_features - generated_features))


def style_loss(style_features, generated_features):
    """
    Style loss measures how well style is transferred.
    
    Uses Gram matrices to compare style representations.
    
    Args:
        style_features: Features from style image
        generated_features: Features from generated image
    
    Returns:
        Style loss value
    """
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)
    
    return tf.reduce_mean(tf.square(style_gram - generated_gram))


class StyleTransferLoss(keras.losses.Loss):
    """
    Combined loss for training style transfer models.
    
    Combines content loss, style loss, and optional total variation loss.
    """
    
    def __init__(
        self,
        encoder,
        content_weight=1.0,
        style_weight=100.0,
        tv_weight=1e-4,
        style_layers=None,
        **kwargs
    ):
        """
        Args:
            encoder: VGG encoder for extracting features
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss (smoothness)
            style_layers: List of layer names for style loss
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Layers to use for style loss (multiple layers capture different scales)
        if style_layers is None:
            self.style_layers = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1'
            ]
        else:
            self.style_layers = style_layers
        
        # Build style feature extractor
        self._build_style_extractor()
    
    def _build_style_extractor(self):
        """Build model to extract features from multiple layers."""
        vgg = keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False
        
        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        
        self.style_extractor = keras.Model(
            inputs=vgg.input,
            outputs=outputs,
            name='style_extractor'
        )
    
    def total_variation_loss(self, image):
        """
        Total variation loss encourages spatial smoothness.
        
        Reduces noise in generated images.
        """
        # Compute differences in x and y directions
        x_diff = image[:, :-1, :-1, :] - image[:, :-1, 1:, :]
        y_diff = image[:, :-1, :-1, :] - image[:, 1:, :-1, :]
        
        return tf.reduce_mean(tf.square(x_diff)) + tf.reduce_mean(tf.square(y_diff))
    
    def call(self, y_true, y_pred):
        """
        Compute total loss.
        
        Args:
            y_true: Tuple of (content_image, style_image)  
            y_pred: Generated stylized image
        
        Returns:
            Total loss value
        """
        # Unpack inputs - y_true is actually just content (target)
        # We'll get content and style from the model inputs
        content_image = y_true
        
        # Content loss (from encoder)
        content_feat = self.encoder(content_image)
        generated_feat = self.encoder(y_pred)
        c_loss = content_loss(content_feat, generated_feat)
        
        # For style loss, we need the style image
        # This is a simplified version - in practice, style comes from model
        # Style loss (from multiple layers)  
        # Note: This is a placeholder - actual implementation needs style input
        s_loss = 0.0
        
        # Total variation loss
        tv_loss = self.total_variation_loss(y_pred)
        
        # Combine losses
        total_loss = (
            self.content_weight * c_loss +
            self.tv_weight * tv_loss
        )
        
        return total_loss
