"""
create_dataset.py

Functionality: This file contains the functions to create a dataset from the images and labels
in order to be able to train the diffusion model. 

This script can create a:
    - Dictionary dataset: A dictionary with the image paths as keys and the pokemon type as values
    in order to be able to do preprocess and create the final dataset in a more customizable way.

    - TensorFlow dataset: A dataset with the images and labels preprocessed and ready to be used to
    train the model. The dara is preprocessed using the preprocessing.py script.
"""

# Imports
# =====================================================================
import re
import json
import pandas as pd
import tensorflow as tf
from src.data.preprocess import img_preprocess, label_preprocess


# =====================================================================
# Dataset creation functions
# =====================================================================


# Datset dictionary TODO: MIRAR NOMBRE Y POSIBLES MEJORAS
# =====================================================================
def dataset_dict(
    image_paths: list, df: pd.DataFrame, save: bool = False, save_path: str = None
) -> dict:
    """Returns a dictionary with the image paths as keys and the pokemon type as values
    so it can be later used to create a dataset

    :param train_paths: The paths to the images
    :param df: The dataframe with the pokemon data
    :param save: Whether to save the dictionary to a file
    :param save_path: The path to save the dictionary to
    :return: A dictionary with the image paths as keys and the pokemon type as values
    """

    data_dict = {}

    for path in image_paths:
        # Associate the image with the pokemon type
        pokemon_id = int(
            re.search(r"\d+", path).group()
        )  # Get the pokemon id from the path

        # Get the pokemon type from the dataframe #
        # TODO: AÑADIR TYPE2
        pokemon_type = df.loc[pokemon_id]["type1"]
        data_dict[path] = pokemon_type

    if save:
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(data_dict, file)
        print(
            f"Dataset dictionary saved to file: {save_path} with {len(data_dict)} entries"
        )

    return data_dict


# Create the dataset TF
# =====================================================================
def dataset_tf(
    dict_dataset: dict = None,
    batch_size: int = 32,
    img_size: int = 64,
    buffer: int = 1,
    save: bool = False,
    save_path: str = None,
    image_paths: list = None,
    df: pd.DataFrame = None,
) -> tf.data.Dataset:
    """Creates a dataset from the image paths and lables and preprocesses them using
    the img_preprocess function and the labels_preprocess function

    :param dataset_dict: The dictionary with the image paths as keys and the pokemon type as values
    :param batch_size: The batch size
    :param img_size: The size to resize the images to
    :param epochs: The number of epochs to repeat the dataset
    :param buffer: The buffer to add to the crop
    :param save: Whether to save the dataset to a file
    :param save_path: The path to save the dataset to
    :param image_paths: The paths to the images
    :param df: The dataframe with the pokemon data
    :return: The dataset divided in images (X_train) and labels (y_train)
    """

    dict_dataset = (
        dataset_dict(image_paths, df) if dict_dataset is None else dict_dataset
    )

    # Get the labels and image paths
    image_paths = list(dict_dataset.keys())
    labels = label_preprocess(
        [label for label in dict_dataset.values()], dict_dataset
    )  # Preprocess the labels to one-hot encoded format

    # Create TensorFlow dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(
        lambda x: img_preprocess(image_path=x, size=img_size, buffer=buffer)
    )  # Preprocess the images

    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # Aplly transformations to the dataset
    dataset = (
        dataset.shuffle(buffer_size=2048)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    if save:
        tf.data.Dataset.save(
            dataset,
            path=save_path,
            compression="GZIP",
        )

        print(f"Dataset saved to file: {save_path}")

    return dataset
