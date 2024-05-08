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


# Main Function
# =====================================================================
def build_unet(img_size: int, num_classes: int) -> tf.keras.Model:
    """Creates the U-Net model

    :param img_size: The size of the image
    :param num_classes: The number of classes
    :return: The U-Net model

    """

    # ----- input -----
    x = x_input = layers.Input(shape=(img_size, img_size, 3), name="x_input")

    x_ts = x_ts_input = layers.Input(shape=(1,), name="x_ts_input")
    x_ts = process_block(x_ts)

    x_label = y_input = layers.Input(shape=(num_classes,), name="y_input")
    x_label = process_block(x_label)

    # ----- left ( down ) -----
    x = x64 = diffusion_block(x, x_ts, x_label)
    x = layers.MaxPool2D(2)(x)

    x = x32 = diffusion_block(x, x_ts, x_label)
    x = layers.MaxPool2D(2)(x)

    x = x16 = diffusion_block(x, x_ts, x_label)
    x = layers.MaxPool2D(2)(x)

    x = x8 = diffusion_block(x, x_ts, x_label)

    # ----- mlp -----
    x = mlp_block(x, x_ts, x_label)

    # ----- right ( up ) -----
    x = layers.Concatenate()([x, x8])
    x = diffusion_block(x, x_ts, x_label)
    x = layers.UpSampling2D(2)(x)

    x = layers.Concatenate()([x, x16])
    x = diffusion_block(x, x_ts, x_label)
    x = layers.UpSampling2D(2)(x)

    x = layers.Concatenate()([x, x32])
    x = diffusion_block(x, x_ts, x_label)
    x = layers.UpSampling2D(2)(x)

    x = layers.Concatenate()([x, x64])
    x = diffusion_block(x, x_ts, x_label)

    # ----- output -----
    x = layers.Conv2D(3, kernel_size=1, padding="same")(x)
    model = tf.keras.models.Model([x_input, y_input, x_ts_input], x)

    return model


# Auxiliary Functions
# =====================================================================
def process_block(input_tensor):
    """
    Process the time steps or label input tensor

    :param input_tensor: The input tensor to process
    :return: The processed tensor

    """

    x = layers.Dense(128, activation="relu")(input_tensor)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def mlp_block(x: tf.Tensor, x_ts: tf.Tensor, x_label: tf.Tensor) -> tf.Tensor:
    """The MLP block of the diffusion model

    :param x: The image to process
    :param x_ts: The time steps to process
    :param x_label: The label to process
    :return: The processed image

    """
    shape = x.shape

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, x_ts, x_label])

    x = layers.Dense(128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(shape[1] * shape[2] * shape[3])(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Reshape(shape[1:])(x)

    return x


def diffusion_block(x_img: tf.Tensor, x_ts: tf.Tensor, x_label: tf.Tensor) -> tf.Tensor:
    """The block of the diffusion model

    :param x_img: The image to process
    :param x_ts: The time steps to process
    :param x_label: The label to process
    :return: The processed image
    """

    x_parameter = layers.Conv2D(128, kernel_size=3, padding="same")(x_img)
    x_parameter = layers.Activation("relu")(x_parameter)

    time_parameter = layers.Dense(128)(x_ts)
    time_parameter = layers.Activation("relu")(time_parameter)
    time_parameter = layers.Reshape((1, 1, 128))(time_parameter)

    label_parameter = layers.Dense(128)(x_label)
    label_parameter = layers.Activation("relu")(label_parameter)
    label_parameter = layers.Reshape((1, 1, 128))(label_parameter)

    x_parameter = x_parameter * label_parameter + time_parameter

    # -----
    x_out = layers.Conv2D(128, kernel_size=3, padding="same")(x_img)
    x_out = x_out + x_parameter
    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation("relu")(x_out)

    return x_out


# def get_sinusoidal_time_embedding(timesteps, embedding_dim):
#     """Get the sinusoidal time embeddings

#     Args:
#         timesteps: The number of timesteps
#         embedding_dim: The embedding dimension

#     Returns:
#         embeddings: The sinusoidal time embeddings
#     """
#     position = tf.range(timesteps, dtype=tf.float32)[:, tf.newaxis]
#     div_term = tf.exp(
#         tf.range(0, embedding_dim, 2, dtype=tf.float32)
#         * -(np.log(10000.0) / embedding_dim)
#     )

#     embeddings = tf.concat(
#         [
#             tf.sin(position * div_term),
#             tf.cos(position * div_term)[:, : embedding_dim // 2],
#         ],
#         axis=1,
#     )
#     return embeddings


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
