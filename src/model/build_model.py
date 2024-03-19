"""
build_model.py

Functionality: This file contains the code to build the diffusion model architecture.

"""

# Imports
# =====================================================================
import configparser
import tensorflow as tf
from tensorflow.keras import layers

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")

IMG_SIZE = int(config["hyperparameters"]["img_size"])
NUM_CLASSES = int(config["hyperparameters"]["num_classes"])


# BuildModel
# =====================================================================
class BuildModel(tf.keras.models.Model):

    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.diffusion_model = self.get_diffusion_model(self.img_size, self.num_classes)

    def get_diffusion_model(
        self, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES
    ) -> tf.keras.models.Model:
        """Creates the diffusion model.

        Args:
            img_size (int, optional): The size of the image. Defaults to IMG_SIZE.
            num_classes (int, optional): The number of classes. Defaults to NUM_CLASSES.

        Returns:
            tf.keras.models.Model: The diffusion model.

        """

        x_input = layers.Input(shape=(img_size, img_size, 3), name="x_input")
        x_ts_input = layers.Input(shape=(1,), name="x_ts_input")
        x_label_input = layers.Input(shape=(num_classes), name="x_label_input")

        x = x_input

        x_ts = layers.Dense(192)(x_ts_input)
        x_ts = layers.LayerNormalization()(x_ts)
        x_ts = layers.Activation("relu")(x_ts)

        x_label = layers.Dense(192)(x_label_input)
        x_label = layers.LayerNormalization()(x_label)
        x_label = layers.Activation("relu")(x_label)

        # ----- left ( down ) -----
        x = x64 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x32 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x16 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x8 = self.block(x, x_ts, x_label)

        # ----- MLP -----
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, x_ts, x_label])
        x = layers.Dense(128)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Dense(8 * 8 * 32)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Reshape((8, 8, 32))(x)

        # ----- right ( up ) -----
        x = layers.Concatenate()([x, x8])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x16])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x32])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x64])
        x = self.block(x, x_ts, x_label)

        # ----- output -----
        x = layers.Conv2D(3, kernel_size=1, padding="same")(x)
        model = tf.keras.models.Model([x_input, x_ts_input, x_label_input], x)
        return model

    def block(self, x_img: tf.Tensor, x_ts: tf.Tensor, x_label: tf.Tensor) -> tf.Tensor:
        """Creates a block of the diffusion model.

        Args:
            x_img (tf.Tensor): The input image tensor.
            x_ts (tf.Tensor): The input time tensor.
            x_label (tf.Tensor): The input label tensor.

        Returns:
            tf.Tensor: The output tensor.
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

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calls the diffusion model.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        return self.diffusion_model(x)


# build_model
# =====================================================================
# def build_model(
#     img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES
# ) -> tf.keras.models.Model:
#     """Builds the diffusion model.

#     Returns:
#         tf.keras.models.Model: The diffusion model.

#     """
#     return BuildModel(img_size, num_classes).diffusion_model
