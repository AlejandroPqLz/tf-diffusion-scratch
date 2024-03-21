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
from src.utils.utils import string_to_onehot, onehot_to_string

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("../../config.ini")

DIR_PATH = config["paths"]["dir_path"]

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

    Attributes:

    - model (tf.keras.Model): The base model to which the diffusion process is added.
    - img_size (int): The size of the input images.
    - num_classes (int): The number of classes in the dataset.
    - T (int): The total number of diffusion steps.
    - beta_start (float): The starting value of beta (noise level).
    - beta_end (float): The ending value of beta (noise level).
    - s (float): The scale factor for the variance curve in the 'cosine' scheduler.
    - scheduler (str): The type of noise schedule ('cosine' or 'linear').

    Methods:

    - train_step(data): The training step for the diffusion model.
    - predict_step(data): The prediction step for the diffusion model.
    - plot_samples(num_samples, poke_type): Generate and plot samples from the diffusion model.
    - forward_diffusion(x_0, t, T, scheduler, beta_start, beta_end, s): Simulate the forward diffusion process.
    - beta_scheduler(scheduler, T, beta_start, beta_end, s): Generate a schedule for beta values according to the specified type.

    """

    def __init__(
        self,
        model: tf.keras.Model,
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
        self.model = model
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
            predicted_noise = self.model([noised_data, t], training=True)
            loss = tf.reduce_mean((target_noise - predicted_noise) ** 2)  # MSE loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------

        # Save the model every 20 epochs
        if self.epoch % 20 == 0:
            self.save(f"{DIR_PATH}/models/inter_models/inter_model_{self.epoch}.h5")

        # Sample and plot a generated image every 10 epochs
        if self.epoch % 10 == 0:
            self.plot_samples(num_samples=1)

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
            predicted_noise = self.model([x_t, normalized_t], training=False)

            # Calculate x_{t-1}
            # 4: x_{t-1} = (x_t - (1 - alpha_t) / sqrt(1 - alpha_cumprod_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z
            sigma_t = tf.sqrt(1 - alpha_cumprod[t])
            x_t = (
                x_t - (1 - alpha[t]) / tf.sqrt(1 - alpha_cumprod[t]) * predicted_noise
            ) / tf.sqrt(alpha[t]) + sigma_t * z

        # 5: end for
        # Return the final denoised image
        return x_t  # 6: return x_0

    def plot_samples(self, num_samples: int = 5, poke_type: int = None):
        """
        Generate and plot samples from the diffusion model.

        Args:
            num_samples (int): The number of samples to generate and plot.
        """

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

        # Generate and plot the samples
        # =====================================================================
        for i in tqdm(range(num_samples), desc="Generating samples", total=num_samples):
            # Start with random noise
            start_noise = tf.random.normal([1, self.img_size, self.img_size, 3])

            y_label = np.zeros(NUM_CLASSES)

            if poke_type is not None:
                poke_type = string_to_onehot(poke_type)
                y_label[poke_type] = 1
            else:
                y_label[np.random.randint(0, NUM_CLASSES - 1)] = 1

            y_label = y_label.reshape(1, NUM_CLASSES)

            # sample = self.predict_step(start_noise, y_label)

            sample = self.predict_step(start_noise)

            # Scale to [0, 1] for plotting
            sample = (sample - tf.reduce_min(sample)) / (
                tf.reduce_max(sample) - tf.reduce_min(sample)
            )

            axs[i].imshow(sample[0])
            axs[i].title.set_text(onehot_to_string(y_label))
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
        beta = DiffusionModel.beta_scheduler(
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
