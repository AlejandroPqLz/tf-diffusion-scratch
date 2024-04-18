"""
diffusion_model.py

Functionality: This file contains the code to define the DiffusionModel class that adds the
diffusion functionality to the defined model.

"""

# Imports
# =====================================================================
import configparser
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.utils import string_to_onehot, onehot_to_string, PROJECT_DIR


# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read(PROJECT_DIR / "config.ini")

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
        self.model = model
        self.img_size = img_size
        self.num_classes = num_classes
        self.T = T
        self.scheduler = scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s

        self.beta = self.beta_scheduler(scheduler, T, beta_start, beta_end, s)
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

        # Rename the variables for easier access
        loss_fn = self.loss
        optimizer = self.optimizer
        T = self.T  # Total diffusion steps
        scheduler = self.scheduler
        beta_start = self.beta_start
        beta_end = self.beta_end
        s = self.s  # Scale factor for the variance curve
        alpha_cumprod = self.alpha_cumprod

        # Unpack the data
        input_data, input_label = data

        # 1: repeat ------

        # 3: t ~ U(0, T)
        # Generate a random timestep for each image in the batch
        t = tf.random.uniform(shape=(), minval=0, maxval=self.T, dtype=tf.int32)
        normalized_t = tf.fill([input_data.shape[0], 1], tf.cast(t, tf.float32) / T)

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
        target_noise = (noised_data - tf.sqrt(alpha_cumprod[t]) * input_data) / tf.sqrt(
            1 - alpha_cumprod[t]
        )

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model(
                [noised_data, normalized_t, input_label], training=True
            )
            loss = loss_fn(target_noise, predicted_noise)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------

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
        alpha = self.alpha
        alpha_cumprod = self.alpha_cumprod

        # Starting from pure noise
        x_t, y_t = data  # 1: x_T ~ N(0, I)

        # Reverse the diffusion process
        # 2: for t = T − 1, . . . , 1 do
        time.sleep(0.4)
        for t in tqdm(reversed(range(1, T)), desc="Sampling sprite", total=T - 1):
            normalized_t = tf.fill([tf.shape(x_t)[0], 1], tf.cast(t, tf.float32) / T)

            # Sample z
            # 3: z ~ N(0, I) if t > 1, else z = 0
            z = tf.random.normal(shape=tf.shape(x_t)) if t > 1 else tf.zeros_like(x_t)

            # Calculate the predicted noise
            predicted_noise = self.model([x_t, normalized_t, y_t], training=False)

            # Calculate x_{t-1}
            # 4: x_{t-1} = (x_t - (1 - alpha_t) / sqrt(1 - alpha_cumprod_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z
            sigma_t = tf.sqrt(1 - alpha_cumprod[t])
            x_t = (
                x_t - (1 - alpha[t]) / tf.sqrt(1 - alpha_cumprod[t]) * predicted_noise
            ) / tf.sqrt(alpha[t]) + sigma_t * z

        # 5: end for
        # Return the final denoised image
        return x_t  # 6: return x_0

    def plot_samples(self, num_samples: int = 5, poke_type: str = None):
        """
        Generate and plot samples from the diffusion model.

        Args:
            num_samples (int): The number of samples to generate and plot.
            poke_type (str): The type of Pokemon to generate samples for. If None, a random type is chosen.
        """

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))

        # TODO: TRATAR LOS TIPOS CUANDO VENGAN VARIOS Y NO SOLO UNO
        if num_samples == 1:
            axs = [axs]  # Make axs iterable when plotting only one sample

        # Generate and plot the samples
        # =====================================================================
        for i in range(num_samples):
            tqdm.write(f"Generating sample {i + 1}/{num_samples}")

            # Start with random noise as input
            start_noise = tf.random.normal([1, self.img_size, self.img_size, 3])

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

        return plt.show()

    @staticmethod
    def forward_diffusion(
        x_0: tf.Tensor,
        t: int,
        T: int,
        scheduler: str,
        beta_start: float,
        beta_end: float,
        s: float,  # **config["training_params"] TODO
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
        alpha = 1.0 - beta
        alpha_cumprod = tf.math.cumprod(alpha)

        # Apply the diffusion process: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
        noise = tf.random.normal(shape=tf.shape(x_0))
        x_t = tf.sqrt(alpha_cumprod[t]) * x_0 + tf.sqrt(1 - alpha_cumprod[t]) * noise

        return x_t

    @staticmethod
    def beta_scheduler(
        scheduler: str, T: int, beta_start: float, beta_end: float, s: float
    ) -> tf.Tensor:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').

        Args:
            scheduler (str): The type of schedule to use. Options are "linear" or "cosine".
            T (int): Total number of timesteps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
            s (float): Scale factor for the variance curve, used in the 'cosine' scheduler.

        Returns:
            tf.Tensor: The beta values for each timestep.
        """

        if scheduler == "linear":
            beta = tf.linspace(beta_start, beta_end, T)

        elif scheduler == "cosine":

            def f(t):  # TODO: CAMBIAR NP POR TF
                return tf.cos((t / T + s) / (1 + s) * tf.constant(np.pi * 0.5)) ** 2

            t = np.arange(0, T + 1)
            alphas_cumprod = f(t) / f(0)
            beta = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            beta = tf.clip_by_value(beta, 0.0001, 0.999)

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        return beta


# Custom Callback for the Diffusion Model
# =====================================================================
class DiffusionCallback(tf.keras.callbacks.Callback):
    """Custom Callback for the Diffusion Model that generates samples every 20 epochs.

    Attributes:
        - diffusion_model (DiffusionModel): The diffusion model to generate samples from.
        - frequency (int): The frequency at which to generate samples.

    Methods:
        - on_epoch_end(epoch, logs): The method that is called at the end of each epoch.

    """

    def __init__(self, diffusion_model: DiffusionModel, frequency=20):
        super(DiffusionCallback, self).__init__()
        # super().__init__()
        self.diffusion_model = diffusion_model
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        """The method that is called at the end of each epoch.

        Args:
            - epoch (int): The current epoch number.
            - logs (dict): The logs containing the training metrics.

        """

        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Generating samples.")
            self.diffusion_model.plot_samples(num_samples=1)
            # self.model.save_weights("diffusion_model.h5")
