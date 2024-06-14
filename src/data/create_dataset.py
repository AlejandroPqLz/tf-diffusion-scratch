"""
create_dataset.py

Functionality: This file contains the functions to create a dataset from the images and labels
in order to be able to train the diffusion model. 

This script can create a:
    - Dictionary dataset: A dictionary with the image paths as keys and the pokemon type as values
    in order to be able to do preprocess and create the final dataset in a more customizable way.

    - TensorFlow dataset: A dataset with the images and labels preprocessed and ready to be used to
    train the model. The data is preprocessed using the preprocessing.py script.
"""

# Imports and setup
# =====================================================================
import re
import json
import logging
from typing import List, Dict
from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.data.preprocess import img_preprocess, label_preprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Dataset creation functions
# =====================================================================
def dataset_tf(
    dict_dataset: Dict[str, str] = None,
    batch_size: int = 32,
    img_size: int = 64,
    buffer: int = 1,
    save: bool = False,
    save_path: Path = Path("./datasets/dataset_tf/"),
    image_paths: List[str] = None,
    df: pd.DataFrame = None,
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from the image paths and labels, preprocessing them using
    the img_preprocess and label_preprocess functions.

    Args:
        dict_dataset (Dict[str, str]): The dictionary with the image paths as keys and the pokemon type as values.
        batch_size (int): The batch size. Defaults to 32.
        img_size (int): The size to resize the images to. Defaults to 64.
        buffer (int): The buffer to add to the crop. Defaults to 1.
        save (bool): Whether to save the dataset to a file. Defaults to False.
        save_path (str): The path to save the dataset to. Required if save is True. Defaults to "./dataset_tf/".
        image_paths (List[str]): The paths to the images.
        df (pd.DataFrame): The dataframe with the pokemon data.

    Returns:
        tf.data.Dataset: The preprocessed dataset ready for training.
    """
    if dict_dataset is None and (image_paths is not None and df is not None):
        dict_dataset = dataset_dict(image_paths, df)

    if dict_dataset is None:
        raise ValueError(
            "Either a dictionary dataset or the image paths and dataframe must be provided."
        )

    # Get the labels and image paths
    image_paths = list(dict_dataset.keys())
    labels = label_preprocess([label for label in dict_dataset.values()], dict_dataset)

    # Create TensorFlow dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(
        lambda x: img_preprocess(image_path=x, size=img_size, buffer=buffer)
    )

    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # Apply transformations to the dataset
    dataset = (
        dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    if save:
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
            logging.info("The path %s did not exist... Path created", save_path)

        tf.data.Dataset.save(dataset, str(save_path), compression="GZIP")
        logging.info("Dataset saved to file: %s", save_path)

    return dataset


# ------


def dataset_dict(
    image_paths: List[str],
    df: pd.DataFrame,
    save: bool = False,
    save_path: Path = Path("./datasets/dataset_dict.json"),
) -> Dict[str, str]:
    """
    Returns a dictionary with the image paths as keys and the pokemon type as values
    so it can be later used to create a dataset.

    Args:
        image_paths (List[str]): The paths to the images.
        df (pd.DataFrame): The dataframe with the pokemon data.
        save (bool): Whether to save the dictionary to a file. Defaults to False.
        save_path (Path): The path to save the dictionary to. Required if save is True.

    Returns:
        Dict[str, str]: A dictionary with the image paths as keys and the pokemon type as values.
    """
    data_dict = {}

    for path in image_paths:
        try:
            # Associate the image with the pokemon type
            pokemon_id = int(re.search(r"\d+", path).group())
            pokemon_type = df.loc[pokemon_id]["type1"]
            data_dict[path] = pokemon_type

        except (KeyError, AttributeError) as e:
            logging.error("Error processing path %s: %s", path, e)

    if save:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info("The path %s did not exist... Path created", save_path.parent)

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f)
            logging.info("Dictionary saved to file: %s", save_path)

    return data_dict
