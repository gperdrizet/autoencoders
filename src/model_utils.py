"""
Model utilities for building, loading, and managing autoencoder models.
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_compression_ae(latent_dim=128, input_shape=(64, 64, 3)):
    """
    Build a convolutional autoencoder for image compression.
    
    Args:
        latent_dim: Dimension of the latent representation
        input_shape: Shape of input images (default: 64x64 for flowers)
    
    Returns:
        Keras Model instance
    """
    # Encoder
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu', name='enc_conv1')(encoder_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu', name='enc_conv2')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', name='enc_conv3')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu', name='enc_conv4')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    encoder = keras.Model(encoder_input, latent, name='encoder')
    
    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    
    x = layers.Dense(4 * 4 * 512, activation='relu')(decoder_input)
    x = layers.Reshape((4, 4, 512))(x)
    
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu', name='dec_conv1')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu', name='dec_conv2')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', name='dec_conv3')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', name='dec_conv4')(x)
    x = layers.BatchNormalization()(x)
    
    decoder_output = layers.Conv2D(3, 3, padding='same', activation='sigmoid', name='decoder_output')(x)
    
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    
    # Full autoencoder
    ae_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, ae_output, name='autoencoder')
    
    return autoencoder, encoder, decoder


def build_anomaly_ae(latent_dim=128, input_shape=(64, 64, 3)):
    """
    Build a convolutional autoencoder for anomaly detection.
    Uses same architecture as compression AE.
    
    Args:
        latent_dim: Dimension of the latent representation
        input_shape: Shape of input images
    
    Returns:
        Keras Model instance (autoencoder only)
    """
    autoencoder, _, _ = build_compression_ae(latent_dim, input_shape)
    return autoencoder


class Sampling(layers.Layer):
    """Reparameterization trick for VAE sampling."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(latent_dim=128, input_shape=(64, 64, 3)):
    """
    Build a Variational Autoencoder (VAE) for image generation.
    
    Args:
        latent_dim: Dimension of the latent space
        input_shape: Shape of input images
    
    Returns:
        Tuple of (vae, encoder, decoder)
    """
    # Encoder
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    
    x = layers.Dense(4 * 4 * 512, activation='relu')(decoder_input)
    x = layers.Reshape((4, 4, 512))(x)
    
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    decoder_output = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    
    # VAE Model
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
            self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
            self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]
        
        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                
                # Reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction),
                        axis=(1, 2)
                    )
                )
                
                # KL divergence loss
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                
                # Total loss (beta-VAE with beta=0.0005 for flowers dataset)
                total_loss = reconstruction_loss + 0.0005 * kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                'loss': self.total_loss_tracker.result(),
                'reconstruction_loss': self.reconstruction_loss_tracker.result(),
                'kl_loss': self.kl_loss_tracker.result(),
            }
        
        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)
    
    vae = VAE(encoder, decoder, name='vae')
    
    return vae, encoder, decoder


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the .keras model file
    
    Returns:
        Loaded Keras model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    
    # Load with custom objects for VAE
    custom_objects = {'Sampling': Sampling}
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load model from {model_path}: {e}')


def quantize_model_float16(model):
    """
    Apply float16 quantization to a model for deployment.
    
    This reduces model size by ~50% with minimal quality loss.
    
    Args:
        model: Keras model to quantize
    
    Returns:
        Quantized model
    """
    # Convert to float16
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    return tflite_model


def save_quantized_model(model, output_path):
    """
    Save a float16 quantized version of the model.
    
    Args:
        model: Keras model to quantize and save
        output_path: Path to save the .tflite model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tflite_model = quantize_model_float16(model)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f'Quantized model saved to {output_path}')


def get_model_info(model):
    """
    Get information about a model.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
    }
