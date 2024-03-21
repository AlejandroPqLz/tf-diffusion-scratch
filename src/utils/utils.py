"""
utils.py

Functionality: This file contains utility functions for the project.
"""

# Imports
# =====================================================================
import configparser
import numpy as np
import pandas as pd
from src.data.preprocess import label_mapping

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")

DATA_PATH = config["paths"]["data_path"]
poke_df = pd.read_csv(f"{DATA_PATH}/raw/pokedex.csv")


# Functions
# =====================================================================
# TODO: Adapt this to the new dataset with the new types (type1 and type2)
def onehot_to_string(one_hot_label: np.ndarray, df: pd.DataFrame = poke_df) -> str:
    """Converts a one-hot encoded label back to a string

    :param one_hot_label: The one-hot encoded label
    :param df: The dataframe with the pokemon data
    :return: The string label
    """

    dict_df = df.set_index("pokedex_id")["type1"].to_dict()

    label_index = np.argmax(one_hot_label)
    for label, index in label_mapping(dict_df).items():
        if index == label_index:
            return label


def string_to_onehot(label: str, df: pd.DataFrame = poke_df) -> np.ndarray:
    """Converts a string label to a one-hot encoded label

    :param label: The string label
    :param df: The dataframe with the pokemon data
    :return: The one-hot encoded label
    """

    dict_df = df.set_index("pokedex_id")["type1"].to_dict()

    label_index = label_mapping(dict_df)[label]
    one_hot = np.zeros(len(dict_df))
    one_hot[label_index] = 1

    return one_hot
