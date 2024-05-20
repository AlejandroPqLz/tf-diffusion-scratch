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
from src.utils.config import Config
from src.utils import CONFIG_PATH


class DiffusionModel(tf.keras.Model):
    """
    DiffusionModel class

    Attributes:

    - model (tf.keras.Model): The base model to which the diffusion process is added.
    - img_size (int): The size of the input images.
    - num_classes (int): The number of classes in the dataset.
    - timesteps(int): The total number of diffusion steps.
    - beta_start (float): The starting value of beta (noise level).
    - beta_end (float): The ending value of beta (noise level).
    - s (float): The scale factor for the variance curve in the 'cosine' scheduler.
    - scheduler (str): The type of noise schedule ('cosine' or 'linear').

    Methods:

    - train_step(data): The training step for the diffusion model.
    - predict_step(data): The prediction step for the diffusion model.
    - plot_samples(num_samples, poke_type): Generate and plot samples from the diffusion model.
    - forward_diffusion(x_0, t, timesteps scheduler, beta_start, beta_end, s): Simulate the forward diffusion process.
    - beta_scheduler(scheduler, timesteps beta_start, beta_end, s): Generate a schedule for beta values according to the specified type.

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

        # Rename the variables for easier access
        loss_fn = self.loss
        optimizer = self.optimizer
        timesteps = self.timesteps  # Total diffusion steps
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
        t = tf.random.uniform(shape=(), minval=0, maxval=timesteps, dtype=tf.int32)
        normalized_t = tf.fill(
            [tf.shape(input_data)[0], 1], tf.cast(t, tf.float32) / timesteps
        )  # TODO: CHECK THIS

        # 2: x_0 ~ q(x_0)
        x_t, x_0, per_noise = self.forward_diffusion(
            input_data,
            t,
            timesteps,
            scheduler,
            beta_start,
            beta_end,
            s,
        )

        alpha_cumprod = tf.cast(alpha_cumprod, tf.float32)

        # 4: eps_t ~ N(0, I)
        target_noise = (x_t - tf.sqrt(alpha_cumprod[t]) * input_data) / tf.sqrt(
            1 - alpha_cumprod[t]
        )  # TODO: CHECK THIS

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model(
                [x_t, input_label, normalized_t], training=True
            )
            loss = loss_fn(target_noise, predicted_noise)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------
        return {"loss": loss}

    def get_last_batch_data(self):
        """
        Get the last batch data for plotting.

        Returns:
            tuple: A tuple containing the input data, target noise, and predicted noise.
        """

        return self.last_batch_data

    def predict_step(self, data):
        """
        Algorithm 2: (sampling) The prediction step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input noised data and label.

        Returns:
            tf.Tensor: The final denoised image.

        """

        # Rename the variables for easier access
        timesteps = self.timesteps
        alpha = self.alpha
        alpha_cumprod = self.alpha_cumprod

        # Starting from pure noise
        x_t, y_t = data  # 1: x_T ~ N(0, I)

        # Reverse the diffusion process
        # 2: for t = T − 1, . . . , 1 do
        time.sleep(0.4)
        for t in tqdm(
            reversed(range(0, timesteps)), desc="Sampling sprite", total=timesteps - 1
        ):
            normalized_t = tf.fill(
                [tf.shape(x_t)[0], 1], tf.cast(t, tf.float32) / timesteps
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
            sigma_t = tf.sqrt(1 - alpha[t])  # TODO: CHECK THIS

            sigma_t = tf.cast(sigma_t, tf.float32)
            alpha = tf.cast(alpha, tf.float32)
            alpha_cumprod = tf.cast(alpha_cumprod, tf.float32)

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
            process (bool): Wether to show the diffusion process or not (every 100 steps). # TODO
        """

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
        """Simulate the forward diffusion process by adding noise to the input image.

        Args:
            x_0 (tf.Tensor): The initial image tensor.
            t (int): The current timestep.
            timesteps(int): The total number of diffusion timesteps.
            scheduler (str): The type of noise schedule ('cosine' or 'linear').
            beta_start (float): The starting value of beta (noise level).
            beta_end (float): The ending value of beta (noise level).
            s (float): The scale factor for the variance curve in the 'cosine' scheduler.

        Returns:
            tf.Tensor: The diffused image tensor at timestep t.
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

        return x_t, x_0, per_noise

    @staticmethod
    def beta_scheduler(
        scheduler: str, timesteps: int, beta_start: float, beta_end: float, s: float
    ) -> tf.Tensor:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').
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


# TODO: PUT THIS IN A SEPARATE FILE
# Custom Callback for the Diffusion Model
# =====================================================================
class PlottingCallback(tf.keras.callbacks.Callback):
    """Custom Callback for the Diffusion Model that plots the input, target noise, and predicted noise.

    Attributes:
        - model (DiffusionModel): The diffusion model to generate samples from.
        - freq (int): The frequency at which to plot the samples.

    Methods:
        - on_epoch_end(epoch, logs): The method that is called at the end of each epoch.

    """

    def __init__(self, diffusion_model, freq=1, img_size=IMG_SIZE):
        super(PlottingCallback, self).__init__()
        self.diffusion_model = diffusion_model
        self.freq = freq  # Frequency to plot during training (every 'freq' epochs)
        self.img_size = img_size

    def on_epoch_end(self, epoch, logs=None):
        """The method that is called at the end of each epoch.

        Args:
            - epoch (int): The current epoch number.
            - logs (dict): The logs containing the training metrics.

        """
        if (epoch + 1) % self.freq == 0:
            x_t, target_noise, predicted_noise, input_data, x_0, per_noise = (
                self.diffusion_model.get_last_batch_data()
            )

            # Normalize the images for plotting
            x_t = (x_t - tf.reduce_min(x_t)) / (tf.reduce_max(x_t) - tf.reduce_min(x_t))
            target_noise = (target_noise - tf.reduce_min(target_noise)) / (
                tf.reduce_max(target_noise) - tf.reduce_min(target_noise)
            )
            predicted_noise = (predicted_noise - tf.reduce_min(predicted_noise)) / (
                tf.reduce_max(predicted_noise) - tf.reduce_min(predicted_noise)
            )

            input_data = (input_data - tf.reduce_min(input_data)) / (
                tf.reduce_max(input_data) - tf.reduce_min(input_data)
            )

            x_0 = (x_0 - tf.reduce_min(x_0)) / (tf.reduce_max(x_0) - tf.reduce_min(x_0))

            per_noise = (per_noise - tf.reduce_min(per_noise)) / (
                tf.reduce_max(per_noise) - tf.reduce_min(per_noise)
            )

            plt.figure(figsize=(10, 5))
            titles = ["Input(noised img)", "Target Noise", "Predicted Noise"]
            for i, data in enumerate([x_t, target_noise, predicted_noise]):
                ax = plt.subplot(1, 3, i + 1)
                ax.imshow(data)  # Plot the first image in the batch
                ax.title.set_text(titles[i])
                ax.axis("off")
            plt.show()

            # get the coordinates of the bottom left corner of the sprite (depends on the sprite size)
            x = 30
            y = 30
            w = 25
            h = 25

            # Slice the tensor to get the pixel values within the background area
            area_noised = x_t[y : y + h, x : x + w, :]
            target_noised = target_noise[y : y + h, x : x + w, :]
            print(
                "MSE area: ", tf.reduce_mean(tf.square(area_noised - target_noised))
            )  # TODO: CHECK THIS

            # Calculate the MSE between the input_noised and the target noise
            img_synthetic = x_t - target_noise
            print(
                "MSE: ", tf.reduce_mean(tf.square(input_data - img_synthetic))
            )  # TODO: CHECK THIS

            _, axs = plt.subplots(2, 3, figsize=(10, 7))
            axs[0, 0].imshow(x_t[y : y + h, x : x + w, :])
            axs[0, 0].set_title("AREA input_noised")

            axs[0, 1].imshow(target_noise[y : y + h, x : x + w, :])
            axs[0, 1].set_title("AREA target noise")

            axs[0, 2].imshow(img_synthetic)
            axs[0, 2].set_title("noised_data - target_noise (synthetic)")

            axs[1, 0].imshow(x_t)
            axs[1, 0].set_title("noised_data (x_t)")

            axs[1, 1].imshow(x_0)
            axs[1, 1].set_title("x_0")

            axs[1, 2].imshow(per_noise)
            axs[1, 2].set_title("per_noise")
            plt.show()


# Custom Callback for the Diffusion Model
# =====================================================================
class DiffusionCallback(tf.keras.callbacks.Callback):
    """Custom Callback for the Diffusion Model that generates samples every 20 epochs.

    Attributes:
        diffusion_model (DiffusionModel): The diffusion model to generate samples from.
        frequency (int): The frequency at which to generate samples.

    Methods:
        on_epoch_end(epoch, logs): The method that is called at the end of each epoch.

    """

    def __init__(self, diffusion_model: DiffusionModel, frequency: int, type: str):
        super(DiffusionCallback, self).__init__()
        self.diffusion_model = diffusion_model
        self.frequency = frequency
        self.type = type

    def on_epoch_end(self, epoch, logs=None):
        """The method that is called at the end of each epoch.

        Args:
            - epoch (int): The current epoch number.
            - logs (dict): The logs containing the training metrics.

        """

        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Generating samples.")
            self.diffusion_model.plot_samples(num_samples=1, poke_type=self.type)
            # self.model.save_weights("diffusion_model.h5")
