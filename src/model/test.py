"""
diffusion_model.py

Functionality: This file contains the code to define the DiffusionModel class that adds the
diffusion functionality to the defined model.

"""

# Imports
# =====================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import configparser
from src.model.build_model import BuildModel

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")

IMG_SIZE = int(config["hyperparameters"]["img_size"])
NUM_CLASSES = int(config["hyperparameters"]["num_classes"])
BATCH_SIZE = int(config["hyperparameters"]["batch_size"])
EPOCHS = int(config["hyperparameters"]["epochs"])
T = int(config["hyperparameters"]["T"])  # Number of diffusion steps
BETA_START = float(config["hyperparameters"]["beta_start"])
BETA_END = float(config["hyperparameters"]["beta_end"])
S = float(config["hyperparameters"]["s"])  # Scale factor for the variance curve
SCHEDULER = config["hyperparameters"]["scheduler"]


class DiffusionModel(tf.keras.Model):
    def __init__(self, img_size: int, num_classes: int):
        super(DiffusionModel, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.diffusion_model = BuildModel(img_size, num_classes)

    def train_step(self, data):
        input_data, _ = data  # Unpacking the data; labels are not used in this context

        # Configurations from the config file or as model attributes
        T = self.T  # Total diffusion steps
        scheduler = self.scheduler
        beta_start = self.beta_start
        beta_end = self.beta_end
        s = self.s  # Scale factor for the variance curve

        with tf.GradientTape() as tape:
            # Generate a random timestep for each image in the batch
            t = tf.random.uniform(shape=(), minval=0, maxval=T, dtype=tf.int32)

            # Perform forward diffusion to get noised images and target noise
            noised_data = self.forward_diffusion(
                input_data, t, T, scheduler, beta_start, beta_end, s
            )
            beta = self.beta_scheduler(scheduler, T, beta_start, beta_end, s)
            alpha_cumprod = np.cumprod(1 - beta)
            target_noise = (
                noised_data - input_data * tf.sqrt(alpha_cumprod[t])
            ) / tf.sqrt(1 - alpha_cumprod[t])

            # Predict noise using the model
            predicted_noise = self.diffusion_model(
                [
                    noised_data,
                    tf.fill([tf.shape(input_data)[0], 1], tf.cast(t, tf.float32) / T),
                ],
                training=True,
            )

            # Compute loss (Mean Squared Error between target and predicted noise)
            loss = tf.reduce_mean((target_noise - predicted_noise) ** 2)

        # Compute gradients and update model weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return training metrics
        return {"loss": loss}

    def predict_step(self, data):
        # Starting from pure noise
        start_noise = (
            data  # Assuming data is the starting noise for the reverse process
        )
        T = self.T  # Total diffusion steps
        scheduler = self.scheduler
        beta_start = self.beta_start
        beta_end = self.beta_end
        s = self.s  # Scale factor for the variance curve

        # Reverse the diffusion process
        for t in reversed(range(T)):
            normalized_t = tf.fill(
                [tf.shape(start_noise)[0], 1], tf.cast(t, tf.float32) / T
            )

            # Update the noise using the model's prediction (this part depends on your model's specific implementation)
            predicted_noise = self.diffusion_model(
                [start_noise, normalized_t], training=False
            )

            # Reverse diffusion step to denoise (this equation is illustrative and should be adapted to your model)
            beta_t = beta_scheduler(scheduler, T, beta_start, beta_end, s)[t]
            alpha_t = 1 - beta_t
            alpha_cumprod_t = np.prod(
                1 - beta_scheduler(scheduler, T, beta_start, beta_end, s)[: t + 1]
            )
            start_noise = (
                start_noise - tf.sqrt(1 - alpha_cumprod_t) * predicted_noise
            ) / tf.sqrt(alpha_t)

        # Return the final denoised image
        return start_noise

    def plot_samples(
        self,
        num_samples: int = 5,
        T: int = None,
        scheduler: str = None,
        beta_start: float = None,
        beta_end: float = None,
        s: float = None,
    ):
        T = T or self.T
        scheduler = scheduler or self.scheduler
        beta_start = beta_start or self.beta_start
        beta_end = beta_end or self.beta_end
        s = s or self.s

        fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

        for i in range(num_samples):
            # Start with random noise
            start_noise = tf.random.normal([1, self.img_size, self.img_size, 3])

            # Reverse the diffusion process to generate an image
            for t in reversed(range(T)):
                normalized_t = tf.fill([1, 1], tf.cast(t, tf.float32) / T)
                predicted_noise = self.diffusion_model(
                    [start_noise, normalized_t], training=False
                )
                beta_t = self.beta_scheduler(scheduler, T, beta_start, beta_end, s)[t]
                alpha_t = 1 - beta_t
                alpha_cumprod_t = np.prod(
                    1
                    - self.beta_scheduler(scheduler, T, beta_start, beta_end, s)[
                        : t + 1
                    ]
                )
                start_noise = (
                    start_noise - tf.sqrt(1 - alpha_cumprod_t) * predicted_noise
                ) / tf.sqrt(alpha_t)

            # Plot the generated image
            img = start_noise[0].numpy()
            img = (img - img.min()) / (
                img.max() - img.min()
            )  # Normalize to [0, 1] for displaying
            axs[i].imshow(img)
            axs[i].axis("off")

        plt.show()

    @staticmethod
    def forward_diffusion(
        x_0: tf.Tensor,
        t: int,
        T: int,
        scheduler: str,
        beta_start: float,
        beta_end: float,
        s: float,
    ) -> tf.Tensor:
        """Simulate the forward diffusion process by adding noise to the input image.

        Args:
            x_0 (tf.Tensor): The initial image tensor.
            t (int): The current timestep.
            T (int): The total number of diffusion timesteps.
            scheduler (str): The type of noise schedule ('cosine' or 'linear').
            beta_start (float): The starting value of beta (noise level).
            beta_end (float): The ending value of beta (noise level).
            s (float): The scale factor for the variance curve in the 'cosine' scheduler.

        Returns:
            tf.Tensor: The diffused image tensor at timestep t.
        """
        # Calculate the noise schedule for beta values
        beta = beta_scheduler(
            scheduler=scheduler, T=T, beta_start=beta_start, beta_end=beta_end, s=s
        )

        # Calculate the cumulative product of (1-beta) to simulate the diffusion process
        alpha = 1.0 - beta
        alpha_cumprod = np.cumprod(alpha)

        # Apply the diffusion process: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
        noise = tf.random.normal(shape=tf.shape(x_0))
        x_t = tf.sqrt(alpha_cumprod[t]) * x_0 + tf.sqrt(1 - alpha_cumprod[t]) * noise

        return x_t

    @staticmethod
    def beta_scheduler(
        scheduler: str, T: int, beta_start: float, beta_end: float, s: float
    ) -> np.array:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').

        Args:
            scheduler (str): The type of schedule to use. Options are "linear" or "cosine".
            T (int): Total number of timesteps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
            s (float): Scale factor for the variance curve, used in the 'cosine' scheduler.

        Returns:
            np.array: An array of beta values according to the selected schedule.
        """
        if scheduler == "linear":
            # Linear schedule: beta values increase linearly from beta_start to beta_end.
            beta = np.linspace(beta_start, beta_end, T)
        elif scheduler == "cosine":
            # Cosine schedule: beta values follow a cosine curve, which can help in controlling the variance.
            def cosine_beta(t):
                return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

            timesteps = np.arange(0, T)
            alphas = cosine_beta(timesteps)
            alphas_cumprod = np.cumprod(alphas)
            beta = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            beta = np.clip(
                beta, a_min=beta_start, a_max=beta_end
            )  # Ensure beta values are within specified range.
            beta = np.concatenate(
                ([beta_start], beta)
            )  # Include the initial beta value.
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        return beta
