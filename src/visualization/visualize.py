"""
visualize.py

Functionality: This file contains the code to plot the data of the project.
"""

# =====================================================================
# Imports
# =====================================================================
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import pandas as pd
from src.model.diffusion import DiffusionModel
from src.utils.utils import onehot_to_string


# =====================================================================
# Plotting functions
# =====================================================================


# Plot random images from a list or dictionary of image paths
# =====================================================================
def plot_image_paths(image_paths: list, n: int = 6) -> None:
    """Plot n random images from the list or dictionary of image paths

    :param image_paths: A list or dictionary of image paths
    :param n: The number of images to plot
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
    dataset_tf: tf.data.Dataset, df: pd.DataFrame = None, n: int = 6
) -> None:
    """Plots a batch of images with their labels, if the time_steps are given plot them too

    :param dataset_tf: The tensorflow dataset with the images and labels
    :param df: The dataframe with the pokemon data
    :param n: The number of images to plot
    :return: None
    """

    plt.figure(figsize=(20, 3))
    for img_batch, label_batch in dataset_tf.shuffle(buffer_size=1000).take(1):
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(img_batch[i] * 0.5 + 0.5)  # [0, 1] instead of [-1, 1]

            if df is not None:
                plt.title(
                    f"{onehot_to_string(label_batch[i], df)}\nShape: {img_batch[i].shape}"
                )
            else:
                plt.title(f"Shape: {img_batch[i].shape}")

        plt.show()


# Plot forward diffusion pass
# =====================================================================


# plot noise levels
# =====================================================================
def plot_noise_levels(T: int, beta_start: float, beta_end: float, s: float) -> None:
    """Plots the noise levels for linear and cosine beta schedules

    :param T: The number of time_steps
    :param beta_start: The starting value of beta
    :param beta_end: The ending value of beta
    :param s: The scale factor for the variance curve
    """

    # Normalize the diffusion step by T for plotting
    normalized_steps = np.linspace(0, 1, T)  # diffusion step (t/T)

    # Variance scheduler for noise level
    # =====================================================================
    beta_linear = DiffusionModel.beta_scheduler("linear", T, beta_start, beta_end, s)
    beta_cosine = DiffusionModel.beta_scheduler("cosine", T, beta_start, beta_end, s)

    alpha_linear = 1.0 - beta_linear
    alpha_cosine = 1.0 - beta_cosine

    aplha_cumprod_linear = np.cumprod(alpha_linear)
    aplha_cumprod_cosine = np.cumprod(alpha_cosine)

    # Plot each scheduler in subplots
    # =====================================================================
    plt.figure(figsize=(10, 7))

    # Noise levels
    plt.plot(normalized_steps, aplha_cumprod_linear, label="linear")
    plt.plot(normalized_steps, aplha_cumprod_cosine, label="cosine")

    plt.title(r"$\alpha_t$")
    plt.xlabel("diffusion step (t/T)")
    plt.ylabel(r"$\alpha_t$")
    plt.legend()


# # Plot the forward diffusion function in 10 different ascending time_steps for the same image
# # =====================================================================
def plot_forward_diffusion(
    X: tf.Tensor,
    scheduler: str,
    timesteps: int,
    T: int,
    beta_start: float,
    beta_end: float,
    s: float,
) -> None:
    """Plot the forward diffusion function in 10 different ascending time_steps for the same image

    :param X: The image to diffuse
    :param scheduler: The beta scheduler
    :param n_timesteps: The number of time_steps to plot
    :param T: The total number of time_steps
    :param beta_start: The starting value of beta
    :param beta_end: The ending value of beta
    :param s: The scale factor for the variance curve

    :return: The diffused image at a time_step t
    """

    n_timesteps = np.linspace(0, T - 1, timesteps, dtype=np.int32)  # Get the time_steps
    plt.figure(figsize=(20, 7))

    for i, t in enumerate(n_timesteps):
        plt.subplot(1, len(n_timesteps), i + 1)
        plt.imshow(
            np.clip(
                DiffusionModel.forward_diffusion(
                    X, t, T, scheduler, beta_start, beta_end, s
                )
                * 0.5
                + 0.5,
                a_min=0.0,
                a_max=1.0,
            )
        )  # [0, 1] instead of [-1, 1]
        if t == 0:
            plt.title(f"{scheduler}\nt:{t}")
        else:
            plt.title(f"t:{t}")
        plt.axis("off")
    plt.show()
