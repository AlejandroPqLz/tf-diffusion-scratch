"""
config.py

Functionality: This file contains the code to load the configuration of the project
in a type-safe way using Pydantic.
"""

# TODO: DO:
# [hyperparameters]
# img_size = 64, int
# num_classes = 10, int
# batch_size = 64, int
# epochs = 10, int
# timesteps = 10, int
# scheduler = "cosine", str
# beta_start = 0.0, float
# beta_end = 1.0, float
# s = 1.0, float

# for config in config_file:
#     config = config.split(", ")
#     key = config[0]
#     value = config[1]
#     config_type = config[2]
#     value = config_type(value)

# Imports
# =====================================================================
from pathlib import Path
import configparser
from pydantic import BaseModel


class Hyperparameters(BaseModel):
    """
    Hyperparameters class to store the hyperparameters of the model.

    Atributes:
        img_size (int): The size of the images.
        num_classes (int): The number of classes in the dataset.
        batch_size (int): The batch size.
        epochs (int): The number of epochs.
        timesteps (int): The number of timesteps.
        scheduler (str): The scheduler to use.
        beta_start (float): The starting beta value.
        beta_end (float): The ending beta value.
        s (float): The s value.
    """

    img_size: int
    num_classes: int
    batch_size: int
    epochs: int
    timesteps: int
    scheduler: str
    beta_start: float
    beta_end: float
    s: float


class Config(BaseModel):
    """
    Config class to store the configuration of the project.

    Atributes:
        hyperparameters (Hyperparameters): The hyperparameters of the model.

    Methods:
        from_config_file(path: Path) -> Config: Load the configuration from a file.
    """

    hyperparameters: Hyperparameters

    @classmethod
    def from_config_file(cls, path: Path):
        """
        Load the configuration from a file.

        Args:
            path (Path): The path to the configuration file.

        Returns:
            Config: The configuration of the project.
        """
        config = configparser.ConfigParser()
        config.read(path)

        hyperparams = {key: value for key, value in config["hyperparameters"].items()}
        # Convert to appropriate types
        hyperparams_converted = {
            k: (
                int(v)
                if v.isdigit()
                else float(v) if v.replace(".", "", 1).isdigit() else v
            )
            for k, v in hyperparams.items()
        }
        return cls(hyperparameters=Hyperparameters(**hyperparams_converted))
