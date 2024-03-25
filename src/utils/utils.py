"""
utils.py

Functionality: This file contains utility functions for the project.
"""

# Imports
# =====================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Set up
# =====================================================================
# TODO: VIEW THE WAY TO GENERALIZE THIS SO IT WORKS IN ANY FOLDER WITH DIFFERENT DEPTHS
PROJECT_DIR = Path(__file__).parents[2]
DATA_PATH = PROJECT_DIR / "data"

poke_df = pd.read_csv(f"{DATA_PATH}/raw/pokedex.csv")


# Functions
# =====================================================================
def label_mapping(dict_dataset: dict) -> dict:
    """Create a mapping from label strings to integer indices

    :param dict_dataset: Dictionary mapping image paths to label strings
    :return: Dictionary mapping label strings to integer indices
    """

    types = sorted(list(set(dict_dataset.values())))
    return {type_: idx for idx, type_ in enumerate(types)}


# TODO: Adapt this to the new dataset with the new types (type1 and type2)
def onehot_to_string(one_hot_label: tf.Tensor, df: pd.DataFrame = poke_df) -> str:
    """Converts a one-hot encoded label back to a string

    :param one_hot_label: The one-hot encoded label
    :param df: The dataframe with the pokemon data
    :return: The string label
    """

    dict_df = df.set_index("pokedex_id")["type1"].to_dict()

    label_index = tf.argmax(one_hot_label)
    for label, index in label_mapping(dict_df).items():
        if index == label_index:
            return label


def string_to_onehot(label: str, df: pd.DataFrame = poke_df) -> tf.Tensor:
    """Converts a string label to a one-hot encoded label

    :param label: The string label
    :param df: The dataframe with the pokemon data
    :return: The one-hot encoded label
    """

    dict_df = df.set_index("pokedex_id")["type1"].to_dict()
    len_dict = len(set(dict_df.values()))

    label_index = label_mapping(dict_df)[label]
    one_hot = tf.zeros(len_dict)

    one_hot[label_index] = 1

    return one_hot
