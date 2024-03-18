"""
preprocess.py

Functionality: This file contains the functions to preprocess the images and labels for the
in order to be able to create a dataset for the diffusion model.
"""

# Imports
# =====================================================================
import tensorflow as tf


# Preprocess the images
# =====================================================================
def img_preprocess(image_path: str, buffer: int = 1, size: int = 64) -> tf.Tensor:
    """Preprocesses the image by cropping it and adapting it to the generative model input

    :param image_path: The path to the image
    :param buffer: The buffer to add to the crop
    :param size: The size to resize the image to
    :return: The preprocessed image
    """

    # Load the image and convert it to a tensor
    # =====================================================================
    img = tf.io.read_file(image_path)  # Read the image
    img = tf.io.decode_image(
        img, channels=3, expand_animations=False
    )  # Decode the image to a tensor (RGB)
    img = tf.image.convert_image_dtype(
        img, tf.float32
    )  # Convert to float32 and normalize to [0,1]

    # Crop the image
    # =====================================================================
    # Find the bounding box of the image
    non_white_pixels = tf.where(
        tf.reduce_all(img < 1.0, axis=-1)
    )  # Find all non-white pixels
    min_x, max_x = tf.reduce_min(non_white_pixels[:, 1]), tf.reduce_max(
        non_white_pixels[:, 1]
    )  # Find min and max x
    min_y, max_y = tf.reduce_min(non_white_pixels[:, 0]), tf.reduce_max(
        non_white_pixels[:, 0]
    )  # Find min and max y
    min_x, max_x, min_y, max_y = (
        tf.cast(min_x, tf.int32),
        tf.cast(max_x, tf.int32),
        tf.cast(min_y, tf.int32),
        tf.cast(max_y, tf.int32),
    )  # cast to int to avoid dtype mismatch

    # Add buffer and crop the image
    min_x, min_y = tf.maximum(0, min_x - buffer), tf.maximum(0, min_y - buffer)
    max_x, max_y = tf.minimum(tf.shape(img)[1], max_x + buffer), tf.minimum(
        tf.shape(img)[0], max_y + buffer
    )

    cropped_img = img[min_y:max_y, min_x:max_x]

    # Final preprocessing
    # =====================================================================
    tensor_img = tf.image.resize(cropped_img, (size, size))  # Resize the image
    final_img = (tensor_img - 0.5) * 2  # Normalize from [0,1] to [-1,1]

    return final_img


# Preprocess the labels
# =====================================================================
def label_preprocess(labels: list, dict_dataset: dict) -> list:
    """Preprocesses all labels to one-hot encoded format

    :param labels: The labels to preprocess
    :param dict_dataset: Dictionary mapping image paths to label strings
    :return: The preprocessed labels in a one-hot list
    """

    label_mapper = label_mapping(dict_dataset)  # label mapping dictionary

    return [
        tf.one_hot(label_mapper[label], depth=len(label_mapper)) for label in labels
    ]


def label_mapping(dict_dataset: dict) -> dict:
    """Create a mapping from label strings to integer indices

    :param dict_dataset: Dictionary mapping image paths to label strings
    :return: Dictionary mapping label strings to integer indices
    """

    types = sorted(list(set(dict_dataset.values())))
    return {type_: idx for idx, type_ in enumerate(types)}
