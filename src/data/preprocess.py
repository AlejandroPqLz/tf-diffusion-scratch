"""
preprocess.py

Functionality: This file contains the functions to preprocess the images and labels for the
in order to be able to create a dataset for the diffusion model.
"""

# Imports and setup
# =====================================================================
import logging
from typing import List, Dict
import tensorflow as tf
from src.utils.utils import label_mapping

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Preprocess the images
# =====================================================================
def img_preprocess(image_path: str, buffer: int = 1, size: int = 64) -> tf.Tensor:
    """
    Preprocesses the image by cropping it and adapting it to the generative model input.

    Args:
        image_path (str): The path to the image.
        buffer (int): The buffer to add to the crop. Defaults to 1.
        size (int): The size to resize the image to. Defaults to 64.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """

    # Load the image and convert it to a tensor
    # =====================================================================
    img = tf.io.read_file(image_path)
    # From image to RGB tensor:
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    # To float32 and normalize to [0,1]:
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Crop the image
    # =====================================================================
    non_white_pixels = tf.where(tf.reduce_all(img < 1.0, axis=-1))
    min_x, max_x = tf.reduce_min(non_white_pixels[:, 1]), tf.reduce_max(
        non_white_pixels[:, 1]
    )
    min_y, max_y = tf.reduce_min(non_white_pixels[:, 0]), tf.reduce_max(
        non_white_pixels[:, 0]
    )
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
    tensor_img = tf.image.resize(cropped_img, (size, size))
    final_img = (tensor_img - 0.5) * 2  # From [0,1] to [-1,1]

    return final_img


# Preprocess the labels
# =====================================================================
def label_preprocess(
    labels: List[str], dict_dataset: Dict[str, str]
) -> List[tf.Tensor]:
    """
    Preprocesses all labels to one-hot encoded format.

    Args:
        labels (List[str]): The labels to preprocess.
        dict_dataset (Dict[str, str]): Dictionary mapping image paths to label strings.

    Returns:
        List[tf.Tensor]: The preprocessed labels in a one-hot list.
    """
    try:
        label_mapper = label_mapping(dict_dataset)
        preprocessed_labels = [
            tf.one_hot(label_mapper[label], depth=len(label_mapper)) for label in labels
        ]

        logging.info("Labels preprocessed successfully")
        return preprocessed_labels

    except KeyError as e:
        logging.error("Label not found in mapping: %s", e)
        raise

    except Exception as e:
        logging.error("Error preprocessing labels: %s", e)
        raise
