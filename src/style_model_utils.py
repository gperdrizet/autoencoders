"""Model architectures for neural style transfer."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def adaptive_instance_normalization(content_features, style_features, epsilon=1e-5):
    """
    Adaptive Instance Normalization (AdaIN).
    
    Adjusts the mean and variance of content features to match style features.
    
    Args:
        content_features: Content feature maps
        style_features: Style feature maps
        epsilon: Small value to avoid division by zero
    
    Returns:
        Normalized features with style statistics
    """
    # Compute content statistics
    content_mean, content_var = tf.nn.moments(
        content_features, axes=[1, 2], keepdims=True
    )
    
    # Compute style statistics
    style_mean, style_var = tf.nn.moments(
        style_features, axes=[1, 2], keepdims=True
    )
    
    # Normalize content features
    normalized_content = (content_features - content_mean) / tf.sqrt(content_var + epsilon)
    
    # Apply style statistics
    stylized = normalized_content * tf.sqrt(style_var + epsilon) + style_mean
    
    return stylized


class AdaINLayer(layers.Layer):
    """Adaptive Instance Normalization as a Keras layer."""
    
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self, inputs):
        content_features, style_features = inputs
        return adaptive_instance_normalization(
            content_features, style_features, self.epsilon
        )


def build_encoder(input_shape=(256, 256, 3)):
    """
    Build VGG19-based encoder for extracting features.
    
    Uses relu4_1 layer from VGG19 as the feature extractor.
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Keras model (encoder)
    """
    # Load pretrained VGG19
    vgg = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    vgg.trainable = False
    
    # Extract features from relu4_1 (common choice for style transfer)
    layer_name = 'block4_conv1'
    
    encoder = keras.Model(
        inputs=vgg.input,
        outputs=vgg.get_layer(layer_name).output,
        name='encoder'
    )
    
    return encoder


def build_decoder(input_shape=(32, 32, 512)):
    """
    Build decoder to reconstruct images from AdaIN features.
    
    Mirror architecture of VGG encoder but in reverse.
    
    Args:
        input_shape: Shape of encoded features (from relu4_1)
    
    Returns:
        Keras model (decoder)
    """
    inputs = keras.Input(shape=input_shape)
    
    # Decoder mirrors VGG encoder
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(inputs)
    x = layers.UpSampling2D(2)(x)  # 32x32 -> 64x64
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(2)(x)  # 64x64 -> 128x128
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(2)(x)  # 128x128 -> 256x256
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Output layer
    outputs = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    
    decoder = keras.Model(inputs, outputs, name='decoder')
    
    return decoder


def build_style_transfer_model(input_shape=(256, 256, 3)):
    """
    Build complete AdaIN-based style transfer model.
    
    Architecture:
    1. Encode content and style images
    2. Apply AdaIN to align content features with style statistics
    3. Decode to generate stylized image
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        encoder, decoder, full_model
    """
    # Build encoder and decoder
    encoder = build_encoder(input_shape)
    
    # Get output shape from encoder
    encoder_output_shape = encoder.output.shape[1:]
    decoder = build_decoder(encoder_output_shape)
    
    # Full model with two inputs (content and style)
    content_input = keras.Input(shape=input_shape, name='content_input')
    style_input = keras.Input(shape=input_shape, name='style_input')
    
    # Encode both images
    content_features = encoder(content_input)
    style_features = encoder(style_input)
    
    # Apply AdaIN
    adain_layer = AdaINLayer()
    stylized_features = adain_layer([content_features, style_features])
    
    # Decode
    output = decoder(stylized_features)
    
    # Create model
    model = keras.Model(
        inputs=[content_input, style_input],
        outputs=output,
        name='style_transfer'
    )
    
    return encoder, decoder, model
