"""
utils.py

Functionality: This file contains utility functions for the project.
"""

# Imports
# =====================================================================
from typing import Dict
import pandas as pd
import tensorflow as tf
from src.utils import DATA_PATH

# Set global variable
# =====================================================================
poke_df = pd.read_csv(f"{DATA_PATH}/raw/pokedex.csv")


# Functions
# =====================================================================
def label_mapping(dict_dataset: Dict[str, str]) -> Dict[str, int]:
    """
    Create a mapping from label strings to integer indices

    Args:
        dict_dataset (Dict[str, str]): Dictionary mapping image paths to label strings

    Returns:
        Dict[str, int]: Dictionary mapping label strings to integer indices
    """
    types = sorted(list(set(dict_dataset.values())))
    return {type_: idx for idx, type_ in enumerate(types)}


def onehot_to_string(one_hot_label: tf.Tensor, df: pd.DataFrame = poke_df) -> str:
    """
    Converts a one-hot encoded label back to a string

    Args:
        one_hot_label (tf.Tensor): The one-hot encoded label
        df (pd.DataFrame): The dataframe with the pokemon data

    Returns:
        str: The string label
    """
    dict_df = df.set_index("pokedex_id")["type1"].to_dict()
    label_index = (
        tf.argmax(one_hot_label, axis=1).numpy()
        if len(one_hot_label.shape) > 1
        else tf.argmax(one_hot_label)
    )

    for label, index in label_mapping(dict_df).items():
        if tf.reduce_any(index == label_index):
            return label


def string_to_onehot(label: str, df: pd.DataFrame = poke_df) -> tf.Tensor:
    """
    Converts a string label to a one-hot encoded label

    Args:
        label (str): The string label
        df (pd.DataFrame): The dataframe with the pokemon data

    Returns:
        tf.Tensor: The one-hot encoded label
    """
    dict_df = df.set_index("pokedex_id")["type1"].to_dict()
    len_dict = len(set(dict_df.values()))
    label_index = label_mapping(dict_df)[label]

    one_hot = tf.zeros(len_dict, dtype=tf.int32)
    one_hot = tf.tensor_scatter_nd_update(one_hot, [[label_index]], [1])

    return one_hot
