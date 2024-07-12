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
from tensorflow.keras import layers, activations
import numpy as np


# Main Function
# =====================================================================
def build_unet(
    img_size: int,
    num_classes: int,
    initial_channels: int = 64,
    channel_multiplier: list = None,
    has_attention: list = None,
    num_group_norm: int = 16,
    dropout_rate: float = 0.0
):
    """Build the U-Net like architecture with diffusion blocks

    Args:
        img_size (int): The size of the input images
        num_classes (int): The number of classes in the dataset
        initial_channels (int): The number of initial channels. Defaults to 64.
        channel_multiplier (list): The channel multiplier for each block. Defaults to None.
        has_attention (list): Whether to apply attention in each block. Defaults to None.
        num_group_norm (int): The number of groups for group normalization. Defaults to 16.
        dropout_rate (float): The rate of dropout. Defaults to 0.0, no dropout.

    Returns:
        model: The custom U-Net model
    """
    # Defeault values (4 blocks, but less processing for smaller images)
    if channel_multiplier is None:
        channel_multiplier = [1, 2, 4, 4] if img_size >= 64 else [1, 2, 2, 4]
    if has_attention is None:
        has_attention = [False, False, True, True] if img_size >= 64 else [False, True, True, True]

    # Validate inputs
    if len(channel_multiplier) != len(has_attention):
        raise ValueError(
            "channel_multiplier and has_attention must have the same length."
        )

    # ----- Input -----
    inputs = layers.Input(shape=(img_size, img_size, 3), name="x_input")
    labels = layers.Input(shape=(num_classes,), name="y_input")
    timesteps = layers.Input(shape=(), name="t_input")  # broadcasts the scalar

    # ----- Embeddings -----
    emb_mult = channel_multiplier[-2]
    label_emb = input_block(labels, initial_channels * emb_mult, num_group_norm)
    time_emb = SinusoidalTimeEmbeddingLayer(initial_channels * emb_mult)(timesteps)
    time_emb = input_block(time_emb, initial_channels * emb_mult, num_group_norm)

    # ----- Encoder -----
    x = inputs
    skips = []
    channels = [initial_channels * m for m in channel_multiplier]
    for i, (ch, attn) in enumerate(zip(channels, has_attention)):
        pooling = True if i < len(channels) - 1 else False
        if i >= 2: # apply dropout in the last few blocks
            x, skip = encoder_block(x, label_emb, time_emb, ch, attn, pooling, num_group_norm, dropout_rate)
        else:
            x, skip = encoder_block(x, label_emb, time_emb, ch, attn, pooling, num_group_norm, 0.0)
        skips.append(skip)

    # ----- Bottleneck -----
    x = bottleneck_block(x, label_emb, time_emb, channels[-1], num_group_norm, dropout_rate)

    # ----- Decoder -----
    skips.reverse()
    for i, (ch, attn) in enumerate(zip(channels[::-1], has_attention[::-1])):
        upsampling = True if i < len(channels) - 1 else False
        if i <= 2: # apply dropout in the first few blocks
            x = decoder_block(x, skips[i], label_emb, time_emb, ch, attn, upsampling, num_group_norm, dropout_rate)
        else:
            x = decoder_block(x, skips[i], label_emb, time_emb, ch, attn, upsampling, num_group_norm, 0.0)

    # ----- Output -----
    x = layers.GroupNormalization(num_group_norm)(x)
    x = layers.Activation(activations.silu)(x)
    outputs = layers.Conv2D(3, 1, padding="same")(x)
    model = tf.keras.Model(inputs=[inputs, labels, timesteps], outputs=outputs)
    return model


# Auxiliary Classes
# =====================================================================
class SinusoidalTimeEmbeddingLayer(layers.Layer):
    """The sinusoidal time embedding layer

    This layer computes the sinusoidal time embeddings for the time steps.

    Args:
        embedding_dim: The embedding dimension

    Methods:
        call: Compute the sinusoidal time embeddings
    """

    def __init__(self, embedding_dim, **kwargs):
        super(SinusoidalTimeEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.emb = np.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, timesteps):
        """Compute the sinusoidal time embeddings

        Args:
            timesteps: The time steps to process

        Returns:
            embeddings: The computed embeddings
        """
        timesteps = tf.cast(timesteps, dtype=tf.float32)
        # embeddings = timesteps[:, None] * self.emb[None, :] same as below
        embeddings = tf.einsum("i,j->ij", timesteps, self.emb)
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        return embeddings


class SelfAttentionLayer(layers.Layer):
    """The self-attention layer

    This layer computes the self-attention mechanism for the input tensor.

    Args:
        channels: The number of channels
        num_group_norm: The number of groups for group normalization

    Methods:
        call: Compute the self-attention
    """

    def __init__(self, channels, num_group_norm, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.norm = layers.GroupNormalization(num_group_norm)
        self.query = layers.Dense(self.channels)
        self.key = layers.Dense(self.channels)
        self.value = layers.Dense(self.channels)
        self.outputs = layers.Dense(self.channels)

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

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        attended_features = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        out = inputs + self.outputs(attended_features)
        return out


# Auxiliary Functions
# =====================================================================
def input_block(input_tensor: tf.Tensor, embedding_dim: int, num_group_norm: int):
    """
    Process the time steps or label input tensor

    Args:
        input_tensor (tf.Tensor): The input tensor
        embedding_dim (int): The embedding dimension
        num_group_norm (int): The number of groups for group normalization

    Returns:
        x: The processed tensor
    """
    x = layers.Dense(embedding_dim, activation=activations.silu)(input_tensor)
    x = layers.GroupNormalization(num_group_norm)(x)
    x = layers.Activation(activations.silu)(x)
    return x


def encoder_block(
    x: tf.Tensor,
    label_emb: tf.Tensor,
    time_emb: tf.Tensor,
    channels: int,
    attention: bool = False,
    pooling: bool = True,
    num_group_norm: int = 16,
    dropout_rate: float = 0.0
):
    """The encoder block

    Args:
        x (tf.Tensor): The image tensor
        label_emb: The label tensor
        time_emb: The time steps tensor
        channels: The number of channels
        attention: Whether to apply attention or not
        pooling: Whether to apply pooling or not
        num_group_norm: The number of groups for group normalization
        dropout_rate: The rate of dropout

    Returns:
        x: The processed tensor
        x: The skipped tensor
    """
    x = skip = process_block(x, label_emb, time_emb, channels, attention, num_group_norm, dropout_rate)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) if pooling else x
    return x, skip


def bottleneck_block(x, label_emb, time_emb, channels, num_group_norm, dropout_rate):
    """The bottleneck block

    Args:
        x: The image tensor
        label_emb: The label tensor
        time_emb: The time steps tensor
        channels: The number of channels
        num_group_norm: The number of groups for group normalization
        dropout_rate: The rate of dropout

    Returns:
        x: The processed tensor
    """
    x = process_block(x, label_emb, time_emb, channels, attention=True, num_group_norm=num_group_norm, dropout_rate=dropout_rate)
    x = process_block(x, label_emb, time_emb, channels, num_group_norm=num_group_norm, dropout_rate=dropout_rate)
    return x


def decoder_block(x, skip, label_emb, time_emb, channels, attention=False, upsampling=True, num_group_norm=16, dropout_rate=0.0):
    """The decoder block

    Args:
        x: The image tensor
        skip: The skip tensor
        label_emb: The label tensor
        time_emb: The time steps tensor
        channels: The number of channels
        attention: Whether to apply attention or not
        upsampling: Whether to apply upsampling or not
        num_group_norm: The number of groups for group normalization
        dropout_rate: The rate of dropout

    Returns:
        x: The processed tensor
    """
    x = layers.Concatenate()([x, skip])
    x = process_block(x, label_emb, time_emb, channels, attention, num_group_norm, dropout_rate)
    x = layers.UpSampling2D(size=(2, 2))(x) if upsampling else x
    return x


def process_block(x_img, label_emb, time_emb, channels, attention=False, num_group_norm=16, dropout_rate=0.2):
    """The process block of the diffusion model

    Args:
        x_img: The image tensor
        label_emb: The label tensor
        time_emb: The time steps tensor
        channels: The number of channels
        attention: Whether to apply attention or not
        num_group_norm: The number of groups for group normalization
        dropout_rate: The rate of dropout

    Returns:
        x_out: The processed tensor
    """
    x_parameter = layers.Conv2D(channels, 3, padding="same")(x_img)
    x_parameter = layers.GroupNormalization(num_group_norm)(x_parameter)
    x_parameter = layers.Activation(activations.silu)(x_parameter)

    label_parameter = layers.Dense(channels)(label_emb)
    label_parameter = layers.Activation(activations.silu)(label_parameter)
    label_parameter = layers.Reshape((1, 1, channels))(label_parameter)

    time_parameter = layers.Dense(channels)(time_emb)
    time_parameter = layers.Activation(activations.silu)(time_parameter)
    time_parameter = layers.Reshape((1, 1, channels))(time_parameter)

    x_parameter = x_parameter * label_parameter + time_parameter

    x_out = layers.Conv2D(channels, 3, padding="same")(x_img)
    x_out += x_parameter

    if attention:
        x_out = SelfAttentionLayer(channels, num_group_norm)(x_out)

    x_out = layers.GroupNormalization(num_group_norm)(x_out)
    x_out = layers.Activation(activations.silu)(x_out)
    x_out = layers.SpatialDropout2D(dropout_rate)(x_out)

    return x_out
