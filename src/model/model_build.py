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
    num_channels: int = 128,
    embedding_dim: int = 128,
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
    inputs = layers.Input(shape=(img_size, img_size, 3), name="x_input")
    labels = layers.Input(shape=(num_classes,), name="y_input")
    timesteps = layers.Input(shape=(1,), name="t_input")

    # ----- Embeddings -----
    # time_emb = SinusoidalTimeEmbeddingLayer(embedding_dim)(timesteps)
    time_emb = process_block(timesteps, embedding_dim)
    label_emb = process_block(labels, embedding_dim)

    # ----- Encoder -----
    x, s1 = encoder_block(inputs, time_emb, label_emb, num_channels, attention=False)
    x, s2 = encoder_block(x, time_emb, label_emb, num_channels, attention=False)
    x, s3 = encoder_block(x, time_emb, label_emb, num_channels, attention=True)
    x, s4 = encoder_block(
        x, time_emb, label_emb, num_channels, attention=True, pooling=False
    )

    # ----- Bottleneck -----
    x = mlp_block(x, time_emb, label_emb, num_channels)  # TODO: ADDD ATTENTION HERE ???

    # ----- Decoder -----
    x = decoder_block(x, s4, time_emb, label_emb, num_channels, attention=True)
    x = decoder_block(x, s3, time_emb, label_emb, num_channels, attention=True)
    x = decoder_block(x, s2, time_emb, label_emb, num_channels, attention=False)
    x = decoder_block(
        x, s1, time_emb, label_emb, num_channels, attention=False, upsampling=False
    )

    # ----- Output -----
    outputs = layers.Conv2D(3, 1, padding="same")(x)

    # ----- Model -----
    model = tf.keras.Model(inputs=[inputs, labels, timesteps], outputs=outputs)
    return model


# Example
# =====================================================================
# model = build_ddpm_unet(32, 18, num_channels=64, embedding_dim=128)
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


class SelfAttentionLayer(layers.Layer):
    """The self-attention layer"""

    def __init__(self, channels, **kwargs):
        """Initialize the self-attention layer

        Args:
            channels: The number of channels
        """
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.query = layers.Conv2D(self.channels, 1, padding="same")
        self.key = layers.Conv2D(self.channels, 1, padding="same")
        self.value = layers.Conv2D(self.channels, 1, padding="same")
        self.outputs = layers.Conv2D(self.channels, 1, padding="same")

    def call(self, inputs):
        """Compute the self-attention

        Args:
            inputs: The input tensor

        Returns:
            output: The output tensor
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.channels, tf.float32) ** (-0.5)

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        attended_features = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        out = self.outputs(attended_features)
        return out


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


def encoder_block(x, time_emb, label_emb, channels, attention=False, pooling=True):
    """The encoder block

    Args:
        x: The image tensor
        time_emb: The time steps tensor
        label_emb: The label tensor
        channels: The number of channels
        attention: Whether to apply attention or not
        pooling: Whether to apply pooling or not

    Returns:
        x: The processed tensor
        x: The skipped tensor
    """
    x = skip = residual_block(x, time_emb, label_emb, channels, attention)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) if pooling else x
    return x, skip


def decoder_block(
    x, skip, time_emb, label_emb, channels, attention=False, upsampling=True
):
    """The decoder block

    Args:
        x: The image tensor
        skip: The skip tensor
        time_emb: The time steps tensor
        label_emb: The label tensor
        channels: The number of channels
        attention: Whether to apply attention or not
        upsampling: Whether to apply upsampling or not

    Returns:
        x: The processed tensor
    """
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, time_emb, label_emb, channels, attention)
    x = layers.UpSampling2D(size=(2, 2))(x) if upsampling else x
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
        x_out = SelfAttentionLayer(channels)(x_out)

    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation("relu")(x_out)

    return x_out


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
