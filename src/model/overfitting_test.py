"""
diffusion_model.py

Functionality: This file contains the code to define the DiffusionModel class that adds the
diffusion functionality to the defined model.

"""

# Imports
# =====================================================================
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.utils import string_to_onehot, onehot_to_string


class DiffusionModel(tf.keras.Model):
    """
    DiffusionModel class

    Attributes:
        model (tf.keras.Model): The base model to which the diffusion process is added.
        img_size (int): The size of the input images.
        num_classes (int): The number of classes in the dataset.
        timesteps (int): The total number of diffusion steps.
        beta_start (float): The starting value of beta (noise level).
        beta_end (float): The ending value of beta (noise level).
        s (float): The scale factor for the variance curve in the 'cosine' scheduler.
        scheduler (str): The type of noise schedule ('cosine' or 'linear').

    Methods:
        train_step(data): The training step for the diffusion model.
        predict_step(data): The prediction step for the diffusion model.
        plot_samples(num_samples, poke_type): Generate and plot samples from the diffusion model.
        forward_diffusion(x_0, t, timesteps, scheduler, beta_start, beta_end, s): Simulate the forward diffusion process.
        beta_scheduler(scheduler, timesteps, beta_start, beta_end, s): Generate a schedule for beta values according to the specified type.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        img_size: int,
        num_classes: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        s: float,
        scheduler: str,
    ):

        super().__init__()
        self.model = model
        self.img_size = img_size
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.scheduler = scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s

        self.beta = self.beta_scheduler(scheduler, timesteps, beta_start, beta_end, s)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = tf.math.cumprod(self.alpha)

    def train_step(self, data):
        """
        Algorithm 1: The training step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input data and labels.

        Returns:
            dict: A dictionary containing the training loss.
        """
        # Unpack the data
        input_data, input_label = data

        # 1: repeat ------

        # 3: t ~ U(0, T)
        # Generate a random timestep for each image in the batch
        t = tf.random.uniform(shape=(), minval=0, maxval=self.timesteps, dtype=tf.int32)
        normalized_t = tf.fill(
            [tf.shape(input_data)[0], 1], tf.cast(t, tf.float32) / self.timesteps
        )  # TODO: CHECK THIS

        # 2: x_0 ~ q(x_0)
        x_t = self.forward_diffusion(
            input_data,
            t,
            self.timesteps,
            self.scheduler,
            self.beta_start,
            self.beta_end,
            self.s,
        )

        alpha_cumprod = tf.cast(self.alpha_cumprod, tf.float32)

        # 4: eps_t ~ N(0, I)
        target_noise = (x_t - tf.sqrt(alpha_cumprod[t]) * input_data) / tf.sqrt(
            1 - alpha_cumprod[t]
        )  # TODO: CHECK THIS

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model(
                [x_t, input_label, normalized_t], training=True
            )
            loss = self.loss_fn(target_noise, predicted_noise)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------
        return {"loss": loss}

    def predict_step(self, data):
        """
        Algorithm 2: (sampling) The prediction step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input noised data and label.

        Returns:
            tf.Tensor: The final denoised image.
        """
        # Starting from pure noise
        x_t, y_t = data  # 1: x_T ~ N(0, I)

        # Reverse the diffusion process
        # 2: for t = T − 1, . . . , 1 do
        time.sleep(0.4)
        inv_process = reversed(range(0, self.timesteps))
        for t in tqdm(inv_process, "Sampling sprite", self.timesteps - 1):
            normalized_t = tf.fill(
                [tf.shape(x_t)[0], 1], tf.cast(t, tf.float32) / self.timesteps
            )  # TODO: CHECK THIS

            # Sample z
            # 3: z ~ N(0, I) if t > 1, else z = 0
            z = (
                tf.random.normal(shape=tf.shape(x_t)) if t > 1 else tf.zeros_like(x_t)
            )  # TODO: CHECK THIS

            # Calculate the predicted noise
            predicted_noise = self.model([x_t, y_t, normalized_t], training=False)

            # Calculate x_{t-1}
            # 4: x_{t-1} = (x_t - (1 - alpha_t) / sqrt(1 - alpha_cumprod_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z
            sigma_t = tf.sqrt(1 - self.alpha[t])  # TODO: CHECK THIS

            sigma_t = tf.cast(sigma_t, tf.float32)
            alpha = tf.cast(self.alpha, tf.float32)
            alpha_cumprod = tf.cast(self.alpha_cumprod, tf.float32)

            # print(
            #     f"x_t: {x_t.dtype}, alpha: {alpha[t].dtype}, alpha_cumprod: {alpha_cumprod[t].dtype}, predicted_noise: {predicted_noise.dtype}, sigma_t: {sigma_t.dtype}, z: {z.dtype}"
            # )

            x_t = (
                x_t - (1 - alpha[t]) / tf.sqrt(1 - alpha_cumprod[t]) * predicted_noise
            ) / tf.sqrt(
                alpha[t]
            ) + sigma_t * z  # TODO: CHECK THIS

        # 5: end for
        return x_t  # 6: return x_0

    def plot_samples(
        self, num_samples: int = 5, poke_type: str = None, process: bool = False
    ) -> None:
        """
        Generate and plot samples from the diffusion model.

        Args:
            num_samples (int): The number of samples to generate and plot.
            poke_type (str): The type of Pokemon to generate samples for. If None, a random type is chosen.
            process (bool): Whether to show the diffusion process or not (every 100 steps).
        """
        # TODO: ADD PROCCES PLOT SO IT PLOT EVERY 100 STEPS HAVING A SUBPLOT OF TIMESTEPS/100 OR DO A GIF FUNCTION OR BOTH

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))

        if num_samples == 1:
            axs = [axs]  # Make axs iterable when plotting only one sample

        # Generate and plot the samples
        # =====================================================================
        for i in range(num_samples):
            tqdm.write(f"Generating sample {i + 1}/{num_samples}")

            # Start with random noise as input that follows N(0, I)
            start_noise = tf.random.normal(
                shape=(1, self.img_size, self.img_size, 3)
            )  # TODO: CHECK THIS

            # Set the label for the sample(s)
            if poke_type is not None:
                y_label = string_to_onehot(poke_type)
            else:
                random_index = tf.random.uniform(
                    shape=[], minval=0, maxval=NUM_CLASSES, dtype=tf.int32
                )
                y_label = tf.one_hot(random_index, NUM_CLASSES)

            y_label = tf.reshape(y_label, [1, NUM_CLASSES])

            # Generate the sample
            sample = self.predict_step((start_noise, y_label))
            sample = tf.squeeze(sample)  # remove the batch dimension

            # Scale to [0, 1] for plotting
            sample = (sample - tf.reduce_min(sample)) / (
                tf.reduce_max(sample) - tf.reduce_min(sample)
            )

            # Plot the sample
            axs[i].imshow(sample)
            axs[i].title.set_text(onehot_to_string(y_label))
            axs[i].axis("off")
        plt.show()

        return None

    @staticmethod
    def forward_diffusion(
        x_0: tf.Tensor,
        t: int,
        timesteps: int,
        scheduler: str,
        beta_start: float,
        beta_end: float,
        s: float,
    ) -> tf.Tensor:
        """
        Simulate the forward diffusion process by adding noise to the input image.

        Args:
            x_0 (tf.Tensor): The initial image tensor.
            t (int): The current timestep.
            timesteps(int): The total number of diffusion timesteps.
            scheduler (str): The type of noise schedule ('cosine' or 'linear').
            beta_start (float): The starting value of beta (noise level).
            beta_end (float): The ending value of beta (noise level).
            s (float): The scale factor for the variance curve in the 'cosine' scheduler.

        Returns:
            tuple: The diffused image tensor at timestep t, the original image tensor, and the added noise tensor.
        """
        # Calculate the noise schedule for beta values
        beta = DiffusionModel.beta_scheduler(
            scheduler, timesteps, beta_start, beta_end, s
        )
        alpha = 1.0 - beta
        alpha_cumprod = tf.math.cumprod(alpha)
        alpha_cumprod = tf.cast(alpha_cumprod, tf.float32)

        # Apply the diffusion process: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
        noise = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)
        x_t = (
            tf.sqrt(alpha_cumprod[t]) * x_0 + tf.sqrt(1 - alpha_cumprod[t]) * noise
        )  # TODO: CHECK THIS
        per_noise = tf.sqrt(1 - alpha_cumprod[t]) * noise  # TODO: CHECK THIS
        x_0 = (x_t - per_noise) / tf.sqrt(alpha_cumprod[t])  # TODO: CHECK THIS

        return x_t

    @staticmethod
    def beta_scheduler(
        scheduler: str, timesteps: int, beta_start: float, beta_end: float, s: float
    ) -> tf.Tensor:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').

        Args:
            scheduler (str): The type of noise schedule ('linear' or 'cosine').
            timesteps (int): The total number of diffusion timesteps.
            beta_start (float): The starting value of beta (noise level).
            beta_end (float): The ending value of beta (noise level).
            s (float): The scale factor for the variance curve in the 'cosine' scheduler.

        Returns:
            tf.Tensor: The beta schedule.
        """

        if scheduler == "linear":
            beta = beta_start + (beta_end - beta_start) * np.arange(timesteps) / (
                timesteps - 1
            )
            # beta = tf.linspace(beta_start, beta_end, timesteps)

        elif scheduler == "cosine":

            def f(t):
                return (
                    tf.cos((t / timesteps + s) / (1 + s) * tf.constant(np.pi * 0.5))
                    ** 2
                )

            t = tf.range(timesteps, dtype=tf.float32)
            alphas_cumprod = f(t) / f(0)
            beta = 1 - alphas_cumprod[1:] / tf.maximum(alphas_cumprod[:-1], 0.999)
            beta = tf.clip_by_value(beta, 0.0001, 0.999)

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        return beta
