"""
build_unet.py

Functionality: This file contains the code to build the diffusion model architecture.
This architecture will be a U-Net like architecture with a diffusion block that processes the
input image with the label and time parameters.

x = f(x, y, t) ,, x = input image, y = label, t = time.

f(x, y, t) = x * y + t.

"""

# Imports
# =====================================================================
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Main Function
# =====================================================================
def build_unet(
    img_size: int,
    num_classes: int,
    num_channels: int = 64,
    embedding_dim: int = 32,
):
    """Creates the U-Net model

    Args:
        img_size (int): The size of the input image
        num_classes (int): The number of classes
        num_channels (int): The number of channels

    Returns:
        model: The U-Net model

    """

    # ----- Input -----
    inputs = layers.Input(shape=(img_size, img_size, 3))
    labels = layers.Input(shape=(num_classes,))
    timesteps = layers.Input(shape=(1,))

    # ----- Embeddings -----
    time_emb = SinusoidalTimeEmbeddingLayer(embedding_dim)(timesteps)
    label_emb = process_block(labels, embedding_dim)

    # ----- Encoder -----
    x = s1 = encoder_block(inputs, time_emb, label_emb, num_channels, attention=False)
    x = s2 = encoder_block(x, time_emb, label_emb, num_channels * 2, attention=True)
    x = s3 = encoder_block(x, time_emb, label_emb, num_channels * 4, attention=True)
    x = s4 = encoder_block(x, time_emb, label_emb, num_channels * 8, attention=False)

    # ----- Bottleneck -----
    x = mlp_block(x, time_emb, label_emb, num_channels * 8)

    # ----- Decoder -----
    x = decoder_block(x, s4, time_emb, label_emb, num_channels * 8, attention=False)
    x = decoder_block(x, s3, time_emb, label_emb, num_channels * 4, attention=True)
    x = decoder_block(x, s2, time_emb, label_emb, num_channels * 2, attention=True)
    x = decoder_block(x, s1, time_emb, label_emb, num_channels, attention=False)

    # ----- Output -----
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(x)

    # ----- Model -----
    model = tf.keras.Model(inputs=[inputs, labels, timesteps], outputs=outputs)
    return model


# Example
# =====================================================================
# model = build_ddpm_unet(32, 18, num_channels=64, time_embedding_dim=128)
# model.summary()


# Auxiliary Classes
# =====================================================================
class SinusoidalTimeEmbeddingLayer(layers.Layer):
    """The sinusoidal time embedding layer"""

    def __init__(self, embedding_dim, **kwargs):
        """Initialize the sinusoidal time embedding layer

        Args:
            embedding_dim: The embedding dimension
        """
        super(SinusoidalTimeEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def call(self, timesteps):
        """Compute the sinusoidal time embeddings

        Args:
            timesteps: The time steps to process

        Returns:
            embeddings: The computed embeddings
        """
        # Ensure timesteps are integers for tf.range
        timesteps = tf.cast(timesteps, dtype=tf.int32)
        timesteps = tf.squeeze(
            timesteps, axis=-1
        )  # Make sure it's the correct shape for tf.range
        position = tf.range(tf.shape(timesteps)[0], dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.embedding_dim, 2, dtype=tf.float32)
            * -(np.log(10000.0) / self.embedding_dim)
        )

        embeddings = tf.concat(
            [
                tf.sin(position * div_term),
                tf.cos(position * div_term)[:, : self.embedding_dim // 2],
            ],
            axis=1,
        )
        return embeddings

    def compute_output_shape(self, input_shape):
        """Compute the output shape

        Args:
            input_shape: The input shape

        Returns:
            shape: The output shape
        """
        return (input_shape[0], self.embedding_dim)


# def self_attention_block(x_img, channels):
#     """The self-attention block

#     Args:
#         x_img: The image tensor
#         channels: The number of channels

#     Returns:
#         attended_features: The attended features
#     """
#     query = layers.Conv2D(channels, 1, padding="same")(x_img)
#     key = layers.Conv2D(channels, 1, padding="same")(x_img)
#     value = layers.Conv2D(channels, 1, padding="same")(x_img)

#     # Calcular la atención
#     scores = tf.einsum("bijc,bjkc->bikc", query, key)
#     scores = tf.nn.softmax(scores)

#     attended_features = tf.einsum("bikc,bjkc->bijc", scores, value)
#     return layers.Conv2D(channels, 1, padding="same")(attended_features)


class SelfAttentionLayer(layers.Layer):
    """The self-attention layer"""

    def __init__(self, channels, **kwargs):
        """Initialize the self-attention layer

        Args:
            channels: The number of channels
        """
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channels = channels

    def build(self):
        """Build the self-attention layer"""
        self.query_conv = layers.Conv2D(self.channels, 1, padding="same")
        self.key_conv = layers.Conv2D(self.channels, 1, padding="same")
        self.value_conv = layers.Conv2D(self.channels, 1, padding="same")
        self.output_conv = layers.Conv2D(self.channels, 1, padding="same")

    def call(self, inputs):
        """Compute the self-attention

        Args:
            inputs: The input tensor

        Returns:
            output: The output tensor
        """
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        scores = tf.matmul(query, key, transpose_b=True)
        distribution = tf.nn.softmax(scores)
        attention_output = tf.matmul(distribution, value)
        output = self.output_conv(attention_output)
        return output

    def compute_output_shape(self, input_shape):
        """Compute the output shape

        Args:
            input_shape: The input shape

        Returns:
            shape: The output shape
        """
        return input_shape


# Auxiliary Functions
# =====================================================================
def process_block(input_tensor, embedding_dim):
    """
    Process the time steps or label input tensor

    Args:
        input_tensor: The input tensor
        embedding_dim: The embedding dimension

    Returns:
        x: The processed tensor
    """
    x = layers.Dense(embedding_dim, activation="relu")(input_tensor)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def mlp_block(
    x: tf.Tensor, time_emb: tf.Tensor, label_emb: tf.Tensor, channels: int
) -> tf.Tensor:
    """The MLP block of the diffusion model

    Args:
        x: The image to process
        time_emb: The time steps to process
        label_emb: The label to process
        channels: The number of channels

    Returns:
        x: The processed image
    """
    shape = x.shape

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, time_emb, label_emb])

    x = layers.Dense(channels)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(shape[1] * shape[2] * shape[3])(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Reshape(shape[1:])(x)

    return x


def residual_block(x_img, time_emb, label_emb, channels, attention=False):
    """The residual block of the diffusion model

    Args:
        x_img: The image tensor
        time_emb: The time steps tensor
        label_emb: The label tensor
        channels: The number of channels
        attention: Whether to apply attention or not

    Returns:
        x_out: The processed tensor
    """
    x_parameter = layers.Conv2D(channels, 3, padding="same")(x_img)
    x_parameter = layers.Activation("relu")(x_parameter)

    time_parameter = layers.Dense(channels)(time_emb)
    time_parameter = layers.Activation("relu")(time_parameter)
    time_parameter = layers.Reshape((1, 1, channels))(time_parameter)

    label_parameter = layers.Dense(channels)(label_emb)
    label_parameter = layers.Activation("relu")(label_parameter)
    label_parameter = layers.Reshape((1, 1, channels))(label_parameter)

    x_parameter = x_parameter * label_parameter + time_parameter

    x_out = layers.Conv2D(channels, 3, padding="same")(x_img)
    x_out += x_parameter

    if attention:
        # x_out = self_attention_block(x_out, channels)
        x_out = SelfAttentionLayer(channels)(x_out)

    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation("relu")(x_out)

    return x_out


def encoder_block(x, time_emb, label_emb, channels, attention=False):
    """The encoder block

    Args:
        x: The image tensor
        time_emb: The time steps tensor
        label_emb: The label tensor
        channels: The number of channels
        attention: Whether to apply attention or not

    Returns:
        x: The processed tensor
        x: The skipped tensor
    """
    x = residual_block(x, time_emb, label_emb, channels, attention)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def decoder_block(x, skip, time_emb, label_emb, channels, attention=False):
    """The decoder block

    Args:
        x: The image tensor
        skip: The skip tensor
        time_emb: The time steps tensor
        label_emb: The label tensor
        channels: The number of channels
        attention: Whether to apply attention or not

    Returns:
        x: The processed tensor
    """
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, time_emb, label_emb, channels, attention)
    x = layers.UpSampling2D(size=(2, 2))(x)
    return x
