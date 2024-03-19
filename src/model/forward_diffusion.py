"""
forward_diffusion.py

Functionality: This file contains the code to perform forward diffusion on an image.

"""

# Imports
# =====================================================================
import numpy as np
import tensorflow as tf


# Forward diffusion
# =====================================================================
def forward_diffusion(
    x_0: tf.Tensor,
    t: int,
    T: int,
    scheduler: str,
    beta_start: float,
    beta_end: float,
    s: float,
) -> tf.Tensor:
    """Forward diffusion function

    :param x_0: The image to diffuse (x_0)
    :param T: The total number of timesteps
    :param t: The time_step
    :param scheduler: The beta scheduler
    :param beta_start: The starting value of beta
    :param beta_end: The ending value of beta
    :param s: scale factor for the variance curve
    :return: The diffused image
    """
    # Get schedulers
    beta = beta_scheduler(scheduler, T, beta_start, beta_end, s)
    alpha = 1.0 - beta
    alpha_cumprod = np.cumprod(alpha)

    # Get the diffused image
    noise = tf.random.normal(shape=tf.shape(x_0))
    x_t = np.sqrt(alpha_cumprod[t]) * x_0 + np.sqrt(1 - alpha_cumprod[t]) * noise

    return x_t


def beta_scheduler(
    scheduler: str,
    T: int,
    beta_start: float,
    beta_end: float,
    s: float,
) -> np.array:
    """
    Generates a linear, quadratic or a cosine noise schedule for the given number of timesteps.

    :param scheduler: The type of schedule to use. Options are "linear" or "cosine".
    :param T: Total number of timesteps
    :param beta_start: Starting value of beta
    :param beta_end: Ending value of beta
    :param s: scale factor for the variance curve
    :return: An array of beta values according to the selected schedule.
    """

    if scheduler == "linear":
        beta = np.linspace(beta_start, beta_end, T)

    elif scheduler == "cosine":

        def f(t):
            return np.cos((t / T + s) / (1 + s) * np.pi * 0.5) ** 2

        t = np.linspace(0, T, T + 1)
        alphas_cumprod = f(t) / f(0)
        beta = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        beta = tf.clip_by_value(beta, 0.0001, 0.999)

    return beta
