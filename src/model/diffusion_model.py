"""
diffusion_model.py

Functionality: This file contains the code to define the DiffusionModel class that adds the
diffusion functionality to the defined model.

"""

# Imports
# =====================================================================
import configparser
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model.build_model import BuildModel

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")

DIR_PATH = config["data"]["dir_path"]

IMG_SIZE = int(config["hyperparameters"]["img_size"])
NUM_CLASSES = int(config["hyperparameters"]["num_classes"])
BATCH_SIZE = int(config["hyperparameters"]["batch_size"])
EPOCHS = int(config["hyperparameters"]["epochs"])

TIMESTEPS = int(config["hyperparameters"]["T"])  # Number of diffusion steps
SCHEDULER = config["hyperparameters"]["scheduler"]
BETA_START = float(config["hyperparameters"]["beta_start"])
BETA_END = float(config["hyperparameters"]["beta_end"])
S = float(config["hyperparameters"]["s"])  # Scale factor for the variance curve


class DiffusionModel(tf.keras.Model):
    """
    DiffusionModel class
    """

    def __init__(
        self,
        img_size: int,
        num_classes: int,
        T: int,
        beta_start: float,
        beta_end: float,
        s: float,
        scheduler: str,
    ):

        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.diffusion_model = BuildModel(img_size, num_classes)
        self.T = T
        self.scheduler = scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s

    def train_step(self, data):
        """
        Algorithm 1: The training step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input data and labels.

        Returns:
            dict: A dictionary containing the training loss.
        """

        # Rename the variables for easier access
        T = self.T  # Total diffusion steps
        scheduler = self.scheduler
        beta_start = self.beta_start
        beta_end = self.beta_end
        s = self.s  # Scale factor for the variance curve

        # Unpack the data
        input_data, _ = data

        # Get the scheduler values
        beta = self.beta_scheduler(self.scheduler, T, beta_start, beta_end, s)
        alpha = 1 - beta
        alpha_cumprod = np.cumprod(alpha)

        # 1: repeat ------

        # 3: t ~ U(0, T)
        t = tf.random.uniform(
            shape=(input_data.shape[0], 1), minval=0, maxval=T, dtype=tf.float32
        )  # Generate a random timestep for each image in the batch

        # 2: x_0 ~ q(x_0)
        noised_data = self.forward_diffusion(
            input_data,
            t,
            T,
            scheduler,
            beta_start,
            beta_end,
            s,
        )

        # 4: eps_t ~ N(0, I)
        target_noise = noised_data - input_data * tf.sqrt(alpha_cumprod[t]) / tf.sqrt(
            1 - alpha_cumprod[t]
        )

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            # eps_theta -> model(x_t, t/T)
            predicted_noise = self.diffusion_model([noised_data, t], training=True)
            loss = tf.reduce_mean((target_noise - predicted_noise) ** 2)  # MSE loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------

        # Save the model every 20 epochs
        if self.epoch % 20 == 0:
            self.save(f"{DIR_PATH}/models/inter_models/inter_model_{self.epoch}.h5")

        # Sample and plot a generated image every 10 epochs
        if self.epoch % 10 == 0:
            self.plot_samples(1, T, scheduler, beta_start, beta_end, s)

        # Update and return training metrics
        return {"loss": loss}

    def predict_step(self, data):
        """
        Algorithm 2: (sampling) The prediction step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input data and labels.

        Returns:
            tf.Tensor: The final denoised image.

        """

        # Rename the variables for easier access
        T = self.T  # Total diffusion steps
        scheduler = self.scheduler
        beta_start = self.beta_start
        beta_end = self.beta_end
        s = self.s  # Scale factor for the variance curve

        # Get the scheduler values
        beta = self.beta_scheduler(scheduler, T, beta_start, beta_end, s)
        alpha = 1 - beta
        alpha_cumprod = np.cumprod(alpha)

        # Starting from pure noise
        x_t = data  # 1: x_T ~ N(0, I)

        # Reverse the diffusion process
        # 2: for t = T − 1, . . . , 1 do
        for t in tqdm(reversed(range(1, T)), desc="Sampling", total=T - 1, leave=False):
            normalized_t = tf.fill([tf.shape(x_t)[0], 1], tf.cast(t, tf.float32) / T)

            # Sample z
            # 3: z ~ N(0, I) if t > 1, else z = 0
            z = tf.random.normal(shape=tf.shape(x_t)) if t > 1 else tf.zeros_like(x_t)

            # Calculate the predicted noise
            predicted_noise = self.diffusion_model([x_t, normalized_t], training=False)

            # Calculate x_{t-1}
            # 4: x_{t-1} = (x_t - (1 - alpha_t) / sqrt(1 - alpha_cumprod_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z
            sigma_t = tf.sqrt(1 - alpha_cumprod[t])
            x_t = (
                x_t - (1 - alpha[t]) / tf.sqrt(1 - alpha_cumprod[t]) * predicted_noise
            ) / tf.sqrt(alpha[t]) + sigma_t * z

        # 5: end for
        # Return the final denoised image
        return x_t  # 6: return x_0

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

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

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
