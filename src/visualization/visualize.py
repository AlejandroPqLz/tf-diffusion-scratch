"""
visualize.py

Functionality: This file contains the code to plot the data of the project.
"""

# Imports
# =====================================================================
import random
import configparser
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import pandas as pd
from src.model.diffusion_funcionality import DiffusionModel
from src.model.build_model import build_unet
from src.utils.utils import onehot_to_string
from src.utils.config import parse_config
from src.utils import DATA_PATH, CONFIG_PATH

# Set config file and GPU use
# =====================================================================
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

hyperparameters = parse_config(config, "hyperparameters")

IMG_SIZE = hyperparameters["img_size"]
NUM_CLASSES = hyperparameters["num_classes"]

# ------
gpus_list = tf.config.list_physical_devices("GPU")
gpu = gpus_list[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Create the architecture of the model
u_net = build_unet(IMG_SIZE, NUM_CLASSES)

# Load the pokemon data
poke_df = pd.read_csv(f"{DATA_PATH}/raw/pokedex.csv")


# =====================================================================
# Plotting functions
# =====================================================================


# Plot random images from a list or dictionary of image paths
# =====================================================================
def plot_image_paths(image_paths: Union[List[str], Dict[str, str]], n: int = 6) -> None:
    """
    Plot n random images from a list or dictionary of image paths.

    Args:
        image_paths (Union[List[str], Dict[str, str]]): A list or dictionary of image paths.
        n (int): The number of images to plot. Defaults to 6.

    Raises:
        ValueError: If image_paths is neither a list nor a dictionary.
    """
    plt.figure(figsize=(20, 3))

    if isinstance(image_paths, list):
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(mpimg.imread(image_paths[random.randint(0, len(image_paths))]))
        plt.show()

    elif isinstance(image_paths, dict):
        for i in range(n):
            plt.subplot(1, n, i + 1)
            r = random.choice(list(image_paths.items()))
            img = mpimg.imread(r[0])  # The image path
            plt.imshow(img)
            plt.title(r[1])  # The pokemon type
        plt.show()

    else:
        raise ValueError(
            f"image_paths must be either a list or a dictionary. Got {type(image_paths)} instead."
        )


# Plot a batch of images from a dataset
# =====================================================================
def plot_images_batch(
    dataset_tf: tf.data.Dataset, df: pd.DataFrame = poke_df, n: int = 6
) -> None:
    """
    Plots a batch of images with their labels, if the time_steps are given plot them too.

    Args:
        dataset_tf (tf.data.Dataset): The tensorflow dataset with the images and labels.
        n (int): The number of images to plot. Defaults to 6.
        df (pd.DataFrame): The dataframe with the pokemon data.
    """
    plt.figure(figsize=(20, 3))

    random_pokemon_dataset = dataset_tf.shuffle(buffer_size=len(dataset_tf)).take(1)
    for img_batch, label_batch in random_pokemon_dataset:
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(img_batch[i] * 0.5 + 0.5)  # [0, 1] instead of [-1, 1]
            plt.title(
                f"{onehot_to_string(label_batch[i], df)}\nShape: {img_batch[i].shape}"
            )

        plt.show()


# Plot forward diffusion pass
# =====================================================================
def plot_noise_levels(timesteps: int, beta_start: float, beta_end: float, s: float):
    """
    Plots the noise levels for linear and cosine beta schedules.

    Args:
        timesteps (int): The number of time_steps.
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        s (float): The scale factor for the variance curve.
    """
    normalized_steps = np.linspace(0, 1, timesteps)  # diffusion step (t/T)

    # Create the models for plotting noise levels
    linear_model = DiffusionModel(
        u_net, IMG_SIZE, NUM_CLASSES, timesteps, beta_start, beta_end, s, "linear"
    )
    cosine_model = DiffusionModel(
        u_net, IMG_SIZE, NUM_CLASSES, timesteps, beta_start, beta_end, s, "cosine"
    )

    # Variance scheduler for noise level
    beta_linear = linear_model.beta_scheduler()
    beta_cosine = cosine_model.beta_scheduler()

    alpha_linear = 1.0 - beta_linear
    alpha_cosine = 1.0 - beta_cosine

    aplha_cumprod_linear = tf.math.cumprod(alpha_linear)
    aplha_cumprod_cosine = tf.math.cumprod(alpha_cosine)

    # Plot each scheduler in subplots
    plt.figure(figsize=(10, 7))

    # Noise levels
    plt.plot(normalized_steps, aplha_cumprod_linear, label="linear")
    plt.plot(normalized_steps, aplha_cumprod_cosine, label="cosine")

    plt.title(r"$\alpha_t$")
    plt.xlabel("diffusion step (t/T)")
    plt.ylabel(r"$\alpha_t$")
    plt.legend()


def plot_forward_diffusion(
    img_tensor: tf.Tensor,
    scheduler: str,
    n: int,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    s: float,
):
    """
    Plot the forward diffusion function in 10 different ascending time_steps for the same image.

    Args:
        img_tensor (tf.Tensor): The image to diffuse.
        scheduler (str): The beta scheduler.
        n (int): The number of time_steps to plot.
        timesteps (int): The total number of time_steps.
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        s (float): The scale factor for the variance curve.

    Returns:
        None
    """
    plt.figure(figsize=(20, 7))
    n_timesteps = np.linspace(0, timesteps - 1, n, dtype=np.int32)
    model = DiffusionModel(
        u_net, IMG_SIZE, NUM_CLASSES, timesteps, beta_start, beta_end, s, scheduler
    )

    for i, t in enumerate(n_timesteps):
        plt.subplot(1, len(n_timesteps), i + 1)

        diffused_img_tensor, _ = model.forward_diffusion(img_tensor, t)
        clipped_img = np.clip(diffused_img_tensor * 0.5 + 0.5, a_min=0.0, a_max=1.0)
        plt.imshow(tf.squeeze(clipped_img))

        if t == 0:
            plt.title(f"{scheduler}\nt:{t}")
        else:
            plt.title(f"t:{t}")
        plt.axis("off")
    plt.show()
