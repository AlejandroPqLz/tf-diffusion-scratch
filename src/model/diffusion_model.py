"""
diffusion_model.py

Functionality: This file contains the code to build the diffusion model architecture.

"""

# Imports
# =====================================================================
import configparser
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.model.build_model import BuildModel
from src.model.forward_diffusion import forward_diffusion
from src.model.forward_diffusion import beta_scheduler
from src.visualization.visualize import onehot_to_string

# Set up
# =====================================================================
config = configparser.ConfigParser()
config.read("config.ini")

IMG_SIZE = int(config["hyperparameters"]["img_size"])
NUM_CLASSES = int(config["hyperparameters"]["num_classes"])


# BuildModel
# =====================================================================
class BuildModel(tf.keras.models.Model):

    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.diffusion_model = self.get_diffusion_model(self.img_size, self.num_classes)

    def get_diffusion_model(
        self, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES
    ) -> tf.keras.models.Model:
        """Creates the diffusion model.

        Args:
            img_size (int, optional): The size of the image. Defaults to IMG_SIZE.
            num_classes (int, optional): The number of classes. Defaults to NUM_CLASSES.

        Returns:
            tf.keras.models.Model: The diffusion model.

        """

        x_input = layers.Input(shape=(img_size, img_size, 3), name="x_input")
        x_ts_input = layers.Input(shape=(1,), name="x_ts_input")
        x_label_input = layers.Input(shape=(num_classes), name="x_label_input")

        x = x_input

        x_ts = layers.Dense(192)(x_ts_input)
        x_ts = layers.LayerNormalization()(x_ts)
        x_ts = layers.Activation("relu")(x_ts)

        x_label = layers.Dense(192)(x_label_input)
        x_label = layers.LayerNormalization()(x_label)
        x_label = layers.Activation("relu")(x_label)

        # ----- left ( down ) -----
        x = x64 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x32 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x16 = self.block(x, x_ts, x_label)
        x = layers.MaxPool2D(2)(x)

        x = x8 = self.block(x, x_ts, x_label)

        # ----- MLP -----
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, x_ts, x_label])
        x = layers.Dense(128)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Dense(8 * 8 * 32)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Reshape((8, 8, 32))(x)

        # ----- right ( up ) -----
        x = layers.Concatenate()([x, x8])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x16])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x32])
        x = self.block(x, x_ts, x_label)
        x = layers.UpSampling2D(2)(x)

        x = layers.Concatenate()([x, x64])
        x = self.block(x, x_ts, x_label)

        # ----- output -----
        x = layers.Conv2D(3, kernel_size=1, padding="same")(x)
        model = tf.keras.models.Model([x_input, x_ts_input, x_label_input], x)
        return model

    def block(self, x_img: tf.Tensor, x_ts: tf.Tensor, x_label: tf.Tensor) -> tf.Tensor:
        """Creates a block of the diffusion model.

        Args:
            x_img (tf.Tensor): The input image tensor.
            x_ts (tf.Tensor): The input time tensor.
            x_label (tf.Tensor): The input label tensor.

        Returns:
            tf.Tensor: The output tensor.
        """

        x_parameter = layers.Conv2D(128, kernel_size=3, padding="same")(x_img)
        x_parameter = layers.Activation("relu")(x_parameter)

        time_parameter = layers.Dense(128)(x_ts)
        time_parameter = layers.Activation("relu")(time_parameter)
        time_parameter = layers.Reshape((1, 1, 128))(time_parameter)

        label_parameter = layers.Dense(128)(x_label)
        label_parameter = layers.Activation("relu")(label_parameter)
        label_parameter = layers.Reshape((1, 1, 128))(label_parameter)

        x_parameter = x_parameter * label_parameter + time_parameter

        # -----
        x_out = layers.Conv2D(128, kernel_size=3, padding="same")(x_img)
        x_out = x_out + x_parameter
        x_out = layers.LayerNormalization()(x_out)
        x_out = layers.Activation("relu")(x_out)

        return x_out

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calls the diffusion model.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        return self.diffusion_model(x)


# Diffusion model
# =====================================================================
class DiffusionModel(BuildModel):
    """
    DiffusionModel class
    """

    def __init__(self, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__(img_size, num_classes)
        self.diffusion_model = self.get_diffusion_model(self.img_size, self.num_classes)

    def train(
        self,
        dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.losses.Loss,
        total_epochs: int = 10,
        scheduler: str = "cosine",
        T: int = 100,
    ):
        """Train the model"""

        # Save intermodels by epoch
        # =====================================================================
        prev_epoch = 0
        # TODO

        # Get the scheduler values
        beta = beta_scheduler(scheduler, T, beta_start, beta_end)  # Get beta
        alpha = 1.0 - beta  # Get alpha
        alpha_cumprod = np.cumprod(alpha)  # Get alpha cumulative product

        for epoch in trange(
            prev_epoch,
            total_epochs,
            desc=f"Training",
            total=total_epochs - prev_epoch,
            leave=True,
        ):  # 1: repeat (iterations through the epochs)
            for step, input_data in tqdm(
                enumerate(dataset),
                desc=f"Epoch {epoch+1}/{total_epochs}",
                total=len(dataset),
                leave=True,
            ):  # 1: repeat (iterations through the batches)
                # Generate a single timestep for one entire batch
                t = np.random.randint(0, T)
                normalized_t = np.full(
                    (input_data.shape[0], 1), t / T, dtype=np.float32
                )  # 3: t ~ U(0, T)

                # Get the target noise
                noised_data = forward_diffusion(
                    input_data, t, scheduler
                )  # 2: x_0 ~ q(x_0)
                target_noise = noised_data - input_data * np.sqrt(
                    alpha_cumprod[t]
                )  # 4: eps_t ~ N(0, I)

                # 5: Take a gradient descent step on
                with tf.GradientTape() as tape:
                    predicted_noise = model(
                        [noised_data, normalized_t], training=True
                    )  # eps_theta -> model(x_t, t/T)
                    loss = loss_fn(
                        target_noise, predicted_noise
                    )  # gradient of the loss (MSE(eps_t, eps_theta))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f"EPOCH {epoch+1} LOSS: {loss.numpy():.4f} \n{'='*69}")

            # # Save the model at the end of every 20 epochs
            # if (epoch + 1) % 20 == 0:
            #     print(f"\tSaving model in epoch {epoch+1}")
            #     model.save(f"{folder_epoch}/diffusion_{scheduler}_{epoch+1}.h5")

            # Sample and plot every 10 epochs
            if (epoch + 1) % 10 == 0:
                print("\tSampling images...")
                plot_samples(model, num_samples=3, scheduler=scheduler, T=T)

    def samlping(
        self,
        start_noise: np.ndarray,
        T: int,
        scheduler: str,
        beta_start: float,
        beta_end: float,
    ) -> np.ndarray:
        """
        Samples an image from the model.

        Args:
            start_noise (np.ndarray): The noise to start the sampling from.
            T (int): The number of timesteps to sample for.
            scheduler (str): The type of schedule to use. Options are "linear" or "cosine".
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.

        Returns:
            np.ndarray: The sampled image.
        """

        # Get the beta schedule and corresponding alpha values
        beta = beta_scheduler(scheduler, T, beta_start, beta_end)
        alpha = 1.0 - beta
        alpha_cumprod = np.cumprod(alpha)

        # Set the starting noise
        x_t = start_noise  # 1: x_T ~ N(0, I)

        # Reverse the diffusion process
        for t in tqdm(
            reversed(range(1, T)), desc="Sampling", total=T - 1, leave=False
        ):  # 2: for t = T − 1, . . . , 1 do
            # Compute normalized timestep
            normalized_t = np.array([t / T]).reshape(1, -1).astype("float32")
            # Sample z_t
            z = (
                np.random.normal(size=x_t.shape)
                if t > 1
                else np.zeros(x_t.shape).astype("float32")
            )  # 3: z ∼ N(0, I) if t > 1, else z = 0
            # Calculate x_(t-1)
            predicted_noise = self.diffusion_model.predict(
                [x_t, normalized_t], verbose=0
            )  # Predict the noise estimate using the model = eps_theta
            x_t = (
                x_t - (1 - alpha[t]) / np.sqrt(1 - alpha_cumprod[t]) * predicted_noise
            ) / np.sqrt(alpha[t]) + np.sqrt(
                beta[t]
            ) * z  # 4: x_(t-1) = (x_t - (1 - alpha_t) / sqrt(1 - alpha_cumprod_t) * eps_theta) / sqrt(alpha_t) + sigma_t * z

        # Return the final sample
        return x_t  # 5: return x_0

    def plot_samples(
        self,
        num_samples: int = 2,
        T: int = 1000,
        scheduler: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
    ) -> None:
        """
        Plots samples from the model.

        Args:
            num_samples (int): The number of samples to plot.
            T (int): The number of timesteps to sample for.
            scheduler (str): The type of schedule to use. Options are "linear" or "cosine".
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
        """

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))

        for i in trange(num_samples, desc="Sample plot", leave=True):

            start_noise = np.random.normal(
                size=(1, self.img_size, self.img_size, 3)
            ).astype("float32")
            y_label = self.generate_em().reshape(1, self.num_classes)
            sample = self.sampling(start_noise, T, scheduler, beta_start, beta_end)
            sample = (sample + 1.0) / 2.0  # Scale to [0, 1]
            axs[i].imshow(sample[0])
            axs[i].title.set_text(self.onehot_to_string(y_label[0]))
            axs[i].axis("off")

        plt.show()

    def save(self, model_name: str, path: str):
        """Save the model"""

        self.diffusion_model.save(f"{path}/{model_name}.h5")

        return f"Model saved as {model_name} in {path}."

    def load(self, model_name: str, path: str):
        """Load the model"""

        self.diffusion_model = tf.keras.models.load_model(f"{path}/{model_name}.h5")

        return f"Model {model_name} loaded from {path}."

    # Generate a random embedding (label) =====================================================================
    def generate_em(self, num_classes: int = NUM_CLASSES) -> np.ndarray:
        """
        Generates a random embedding (label)

        Args:
            num_classes (int): The number of classes.

        Returns:
            np.ndarray: The random embedding.
        """
        em = np.zeros(num_classes)
        em[np.random.randint(0, num_classes - 1)] = 1

        return em


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


# =====================================================================


class DiffusionModel(tf.keras.Model):
    """
    DiffusionModel class

    """

    def __init__(self, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

    def train_step(self, data):
        # Unpack the data
        input_data, label = data

        # Compute the scheduler values
        beta = beta_scheduler(scheduler, T, beta_start, beta_end)
        alpha = 1.0 - beta
        alpha_cumprod = np.cumprod(alpha)

        # 1: repeat (iterations through the epochs)

        # 3: t ~ U(0, T)
        t = np.random.randint(0, T)
        normalized_t = np.full((input_data.shape[0], 1), t / T, dtype=np.float32)

        # 2: x_0 ~ q(x_0)
        noised_data = forward_diffusion(input_data, t, scheduler)

        # 4: eps_t ~ N(0, I)
        target_noise = noised_data - input_data * np.sqrt(alpha_cumprod[t])

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self(
                [noised_data, normalized_t], training=True
            )  # # eps_theta -> model(x_t, t/T)
            loss = loss_fn(target_noise, predicted_noise)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss}
