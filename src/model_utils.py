"""
Model architectures for image compression and denoising autoencoders.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_compression_ae(latent_dim=512, input_shape=(256, 256, 3)):
    """
    Build a convolutional autoencoder for image compression.
    
    Architecture:
        Encoder: Conv2D layers progressively reduce spatial dimensions
        Latent: Dense bottleneck layer (ALL information must pass through here)
        Decoder: Conv2DTranspose layers reconstruct from latent only
    
    Args:
        latent_dim: Dimension of compressed representation (default: 512)
        input_shape: Input image shape (H, W, C)
    
    Returns:
        (autoencoder, encoder, decoder) - Full model and component models
    """
    # Calculate compression ratio
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    compression_ratio = input_size / latent_dim
    
    print(f"\nBuilding Compression Autoencoder")
    print(f"  Input shape: {input_shape}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Compression ratio: {compression_ratio:.1f}×")
    
    # ============ ENCODER ============
    encoder_input = layers.Input(shape=input_shape, name='input_image')
    x = encoder_input
    
    # Downsampling blocks with deeper convolutions
    # 256×256 → 128×128
    x = layers.Conv2D(64, 3, strides=2, padding='same', name='enc_conv1')(x)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='enc_conv1b')(x)
    x = layers.BatchNormalization(name='enc_bn1b')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu1b')(x)
    
    # 128×128 → 64×64
    x = layers.Conv2D(128, 3, strides=2, padding='same', name='enc_conv2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu2')(x)
    x = layers.Conv2D(128, 3, padding='same', name='enc_conv2b')(x)
    x = layers.BatchNormalization(name='enc_bn2b')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu2b')(x)
    
    # 64×64 → 32×32
    x = layers.Conv2D(256, 3, strides=2, padding='same', name='enc_conv3')(x)
    x = layers.BatchNormalization(name='enc_bn3')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu3')(x)
    x = layers.Conv2D(256, 3, padding='same', name='enc_conv3b')(x)
    x = layers.BatchNormalization(name='enc_bn3b')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu3b')(x)
    
    # 32×32 → 16×16
    x = layers.Conv2D(512, 3, strides=2, padding='same', name='enc_conv4')(x)
    x = layers.BatchNormalization(name='enc_bn4')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu4')(x)
    x = layers.Conv2D(512, 3, padding='same', name='enc_conv4b')(x)
    x = layers.BatchNormalization(name='enc_bn4b')(x)
    x = layers.LeakyReLU(0.2, name='enc_relu4b')(x)
    
    # Flatten and compress to latent dimension (16×16×512 = 131,072)
    x = layers.Flatten(name='enc_flatten')(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    encoder = keras.Model(encoder_input, latent, name='encoder')
    
    # ============ DECODER ============
    decoder_input = layers.Input(shape=(latent_dim,), name='latent_input')
    x = decoder_input
    
    # Project and reshape (16×16×512 = 131,072)
    x = layers.Dense(16 * 16 * 512, activation='relu', name='dec_dense')(x)
    x = layers.Reshape((16, 16, 512), name='dec_reshape')(x)
    
    # Upsampling blocks (deeper to match encoder)
    # 16×16 → 32×32
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', name='dec_conv1')(x)
    x = layers.BatchNormalization(name='dec_bn1')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu1')(x)
    x = layers.Conv2D(512, 3, padding='same', name='dec_conv1b')(x)
    x = layers.BatchNormalization(name='dec_bn1b')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu1b')(x)
    
    # 32×32 → 64×64
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', name='dec_conv2')(x)
    x = layers.BatchNormalization(name='dec_bn2')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu2')(x)
    x = layers.Conv2D(256, 3, padding='same', name='dec_conv2b')(x)
    x = layers.BatchNormalization(name='dec_bn2b')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu2b')(x)
    
    # 32×32 → 64×64
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', name='dec_conv3')(x)
    x = layers.BatchNormalization(name='dec_bn3')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu3')(x)
    x = layers.Conv2D(128, 3, padding='same', name='dec_conv3b')(x)
    x = layers.BatchNormalization(name='dec_bn3b')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu3b')(x)
    
    # 128×128 → 256×256
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', name='dec_conv4')(x)
    x = layers.BatchNormalization(name='dec_bn4')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu4')(x)
    x = layers.Conv2D(64, 3, padding='same', name='dec_conv4b')(x)
    x = layers.BatchNormalization(name='dec_bn4b')(x)
    x = layers.LeakyReLU(0.2, name='dec_relu4b')(x)
    
    # Final output layer (256×256×64 → 256×256×3)
    reconstructed = layers.Conv2D(3, 3, padding='same', activation='sigmoid', name='output_image')(x)
    
    decoder = keras.Model(decoder_input, reconstructed, name='decoder')
    
    # ============ FULL AUTOENCODER ============
    # All information flows through the latent bottleneck
    ae_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, ae_output, name='autoencoder')
    
    return autoencoder, encoder, decoder


def build_denoising_ae(latent_dim=4096, input_shape=(256, 256, 3)):
    """
    Build a denoising autoencoder.
    
    Uses same architecture as compression AE.
    
    Args:
        latent_dim: Dimension of compressed representation (default: 4096)
        input_shape: Input image shape (H, W, C)
    
    Returns:
        autoencoder model
    """
    # Use same architecture as compression
    autoencoder, _, _ = build_compression_ae(latent_dim=latent_dim, input_shape=input_shape)
    return autoencoder
