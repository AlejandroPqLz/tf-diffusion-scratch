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

# TIMESTEPS = int(config["hyperparameters"]["T"])  # Number of diffusion steps
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
        t = tf.random.uniform(shape=(), minval=0, maxval=T, dtype=tf.int32)
        # t = np.random.randint(0, T)
        normalized_t = tf.fill([tf.shape(input_data)[0], 1], tf.cast(t, tf.float32) / T)

        # 2: x_0 ~ q(x_0)
        x_t, x_0, per_noise = self.forward_diffusion(
            input_data,
            t,
            T,
            scheduler,
            beta_start,
            beta_end,
            s,
        )

        # 4: eps_t ~ N(0, I)
        target_noise = (x_t - tf.sqrt(alpha_cumprod[t]) * input_data) / tf.sqrt(
            1 - alpha_cumprod[t]
        )

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model(
                [x_t, normalized_t, input_label], training=True
            )
            loss = loss_fn(target_noise, predicted_noise)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 6: until convergence ------

        # Store last batch data for plotting
        self.last_batch_data = (
            x_t[0],
            target_noise[0],
            predicted_noise[0],
            input_data[0],
            x_0[0],
            per_noise[0],
        )

        # Update and return training metrics
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
            # sigma_t = tf.sqrt(1 - alpha_cumprod[t])  # TODO: CHECK

            # for all timesteps.
            sigma_t = (
                tf.sqrt((1 - alpha_cumprod[t - 1]) / (1 - alpha_cumprod[t]))
                if t > 1
                else 0
            )
            # if t > 1:
            #     sigma_t = tf.sqrt((1 - alpha_cumprod[t-1]) / (1 - alpha_cumprod[t]))
            # else:
            #     sigma_t = 0

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

            # Start with random noise as input that follows N(0, I)
            start_noise = tf.random.normal(shape=(1, self.img_size, self.img_size, 3))

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
        per_noise = tf.sqrt(1 - alpha_cumprod[t]) * noise
        x_0 = (x_t - per_noise) / tf.sqrt(alpha_cumprod[t])

        return x_t, x_0, per_noise

    @staticmethod
    def beta_scheduler(
        scheduler: str, T: int, beta_start: float, beta_end: float, s: float
    ) -> tf.Tensor:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').
        """

        if scheduler == "linear":
            beta = tf.linspace(beta_start, beta_end, T)

        elif scheduler == "cosine":

            def f(t):
                return tf.cos((t / T + s) / (1 + s) * tf.constant(np.pi * 0.5)) ** 2

            t = tf.range(0, T + 1, dtype=tf.float32)
            alphas_cumprod = f(t) / f(0)
            beta = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            beta = tf.clip_by_value(beta, 0.0001, 0.999)

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        return beta

    # @staticmethod
    # def beta_scheduler(
    #     scheduler: str, T: int, beta_start: float, beta_end: float, s: float
    # ) -> tf.Tensor:
    #     """
    #     Generates a schedule for beta values according to the specified type ('linear' or 'cosine').

    #     Args:
    #         scheduler (str): The type of schedule to use. Options are "linear" or "cosine".
    #         T (int): Total number of timesteps.
    #         beta_start (float): Starting value of beta.
    #         beta_end (float): Ending value of beta.
    #         s (float): Scale factor for the variance curve, used in the 'cosine' scheduler.

    #     Returns:
    #         tf.Tensor: The beta values for each timestep.
    #     """

    #     if scheduler == "linear":
    #         beta = tf.linspace(beta_start, beta_end, T)

    #     elif scheduler == "cosine":

    #         def f(t):  # TODO: CAMBIAR NP POR TF
    #             return tf.cos((t / T + s) / (1 + s) * tf.constant(np.pi * 0.5)) ** 2

    #         t = np.arange(0, T + 1)
    #         alphas_cumprod = f(t) / f(0)
    #         beta = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    #         beta = tf.clip_by_value(beta, 0.0001, 0.999)

    #     else:
    #         raise ValueError(f"Unsupported scheduler: {scheduler}")

    #     return beta


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

            # get the coordinates of the bottom left corner of the sprite
            # if self.img_size == 64:
            #     x = 60
            #     y = 60
            #     w = 50
            #     h = 50
            # elif self.img_size == 32:
            #     x = 30
            #     y = 30
            #     w = 25
            #     h = 25

            x = 30
            y = 30
            w = 25
            h = 25

            # Slice the tensor to get the pixel values within the background area
            area_noised = x_t[y : y + h, x : x + w, :]
            target_noised = target_noise[y : y + h, x : x + w, :]
            print("MSE area: ", tf.reduce_mean(tf.square(area_noised - target_noised)))

            # Calculate the MSE between the input_noised and the target noise
            img_synthetic = x_t - target_noise
            print("MSE: ", tf.reduce_mean(tf.square(input_data - img_synthetic)))

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
        - diffusion_model (DiffusionModel): The diffusion model to generate samples from.
        - frequency (int): The frequency at which to generate samples.

    Methods:
        - on_epoch_end(epoch, logs): The method that is called at the end of each epoch.

    """

    def __init__(self, diffusion_model: DiffusionModel, frequency=20, type=None):
        super(DiffusionCallback, self).__init__()
        # super().__init__()
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
