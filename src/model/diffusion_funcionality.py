"""
diffusion_functionality.py

Functionality: This file contains the code to define the DiffusionModel class that adds the
diffusion functionality to the defined model.

"""

# Imports and setup
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
        compile(loss, optimizer, **kwargs): Compile the model with the specified loss and optimizer.
        train_step(data): The training step for the diffusion model.
        sampling_step(data): The sampling step for the diffusion model.
        forward_diffusion(x_0, t): Diffuse the data by adding noise to the input image.
        beta_scheduler(): Generates a schedule for beta values according to the specified type.
        gather(tensor, index): Extract the value at the specified index from the tensor.
        plot_samples(num_samples, poke_type): Generate and plot samples from the diffusion model.
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
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.scheduler = scheduler

        self.beta = tf.constant(self.beta_scheduler(), tf.float32)
        self.alpha = tf.constant(1 - self.beta, tf.float32)
        self.alpha_cumprod = tf.constant(tf.math.cumprod(self.alpha), tf.float32)
        self.sigma = tf.constant(tf.sqrt(self.beta), tf.float32)

        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def compile(
        self,
        loss: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        **kwargs,
    ):
        """
        Compile the model with the specified loss and optimizer.

        Args:
            loss (tf.keras.losses.Loss): The loss function to use for training.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training.
            **kwargs: Additional arguments to pass to the model's compile method.
        """
        super().compile(**kwargs)
        self.compute_loss = loss
        self.optimizer = optimizer

    def train_step(self, data: tuple) -> dict:
        """
        Algorithm 1: The training step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input data and labels.

        Returns:
            dict: A dictionary containing the updated metrics.
        """
        # Unpack the data
        input_data, input_label = data
        batch_size = tf.shape(input_data)[0]

        # 1: repeat ------

        # 3: t ~ U(0, T): Generate a random timestep for each image in the batch
        t = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.timesteps, dtype=tf.int32
        )
        normalized_t = (
            tf.cast(t, tf.float32) / self.timesteps
        )  # TODO: CHECK IF ITS NECESSARY

        # 2: x_0 ~ q(x_0), 4: ε_t ~ N(0, I)
        x_t, target_noise = self.forward_diffusion(input_data, t)

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model(
                [x_t, input_label, normalized_t], training=True
            )
            loss = self.compute_loss(target_noise, predicted_noise)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 6: until convergence ------

        # Update the metrics # TODO: check this
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target_noise, predicted_noise)

        return {m.name: m.result() for m in self.metrics}

    def sampling_step(self, data: tuple) -> tf.Tensor:
        """
        Algorithm 2: (sampling) The sampling step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input noised data and label.

        Returns:
            tf.Tensor: The final denoised image or the intermediate steps if process is True.
        """
        # 1: x_T ~ N(0, I): Starting from pure noise
        x_t, y_t = data
        batch_size = tf.shape(x_t)[0]

        # 2: for t = T, . . . , 0 do: Reverse the diffusion process
        time.sleep(0.4)
        inv_process = reversed(range(0, self.timesteps))
        for t in tqdm(inv_process, desc="Sampling sprite...", total=self.timesteps):
            normalized_t = (
                tf.cast(t, tf.float32) / self.timesteps
            )  # TODO: CHECK IF ITS NECESSARY
            normalized_t = tf.fill(batch_size, normalized_t)  # shape = (batch_size,)

            # 3: z ~ N(0, I) if t > 0, else z = 0: Sample noise, except at the first timestep
            z = tf.random.normal(shape=tf.shape(x_t)) if t > 0 else tf.zeros_like(x_t)

            # 4: x_{t-1} = (x_t - (1 - α_t) / sqrt(1 - α_cumprod_t) * ε_θ) / sqrt(α_t) + σ_t * z
            predicted_noise = self.model([x_t, y_t, normalized_t], training=False)
            alpha_t = self.gather(self.alpha, t)
            alpha_cumprod_t = self.gather(self.alpha_cumprod, t)
            sigma_t = self.gather(self.sigma, t)

            x_t = (
                x_t - (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t) * predicted_noise
            ) / tf.sqrt(alpha_t) + sigma_t * z

        # 5: end for
        return x_t  # 6: return x_0

    def forward_diffusion(self, x_0: tf.Tensor, t: int) -> tf.Tensor:
        """
        Diffuse the data by adding noise to the input image.

        Args:
            x_0 (tf.Tensor): The initial image tensor.
            t (int): The current timestep.

        Returns:
            tf.Tensor: The diffused image tensor at timestep t.
            noise (tf.Tensor): The noise added to the image.
        """
        noise = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)
        alpha_cumprod_t = self.gather(self.alpha_cumprod, t)

        mean = tf.sqrt(alpha_cumprod_t) * x_0
        variance = 1 - alpha_cumprod_t

        x_t = mean + tf.sqrt(variance) * noise

        return x_t, noise

    def beta_scheduler(self) -> tf.Tensor:
        """
        Generates a schedule for beta values according to the specified type ('linear' or 'cosine').

        Returns:
            tf.Tensor: The beta values for each timestep (the beta scheduler)
        """
        if self.scheduler == "linear":
            scale = 1000 / self.timesteps
            beta_start = self.beta_start * scale
            beta_end = self.beta_end * scale
            beta = tf.linspace(beta_start, beta_end, self.timesteps)

        elif self.scheduler == "cosine":

            def f(t):
                pi = tf.constant(np.pi)
                return (
                    tf.cos((t / self.timesteps + self.s) / (1 + self.s) * (pi * 0.5))
                    ** 2
                )

            t = tf.range(self.timesteps, dtype=tf.float32)
            alphas_cumprod_t = f(t) / f(0)
            alphas_cumprod_t_prev = f(t - 1) / f(0)
            beta_t = 1 - alphas_cumprod_t / alphas_cumprod_t_prev
            beta = tf.clip_by_value(beta_t, 0.0001, 0.999)  # for numerical stability

        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

        return beta

    def gather(self, tensor, index):
        """
        Extract the value at the specified index from the tensor.

        Args:
            tensor (tf.Tensor): The tensor to gather values from.
            index (int): The index to gather values for.

        Returns:
            tf.Tensor: The gathered tensor values.
        """
        tensor_t = tf.gather(tensor, index)
        return tf.reshape(tensor_t, [-1, 1, 1, 1])

    def plot_samples(
        self, num_samples: int = 1, poke_type: str = None, start_noise: tf.Tensor = None
    ) -> None:
        """
        Generate and plot samples from the diffusion model.

        Args:
            num_samples (int): The number of samples to generate and plot.
            poke_type (str): The type of Pokemon to generate samples for.
            If None, a random type is chosen.
            start_noise (tf.Tensor): The starting noise tensor. If None, random noise is used.

        """

        _, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))

        if num_samples == 1:
            axs = [axs]  # Make axs iterable when plotting only one sample

        # Generate and plot the samples
        # =====================================================================
        for i in range(num_samples):
            tqdm.write(f"Generating sample {i + 1}/{num_samples}")

            # Start with random noise as input that follows N(0, I)
            if start_noise is None:
                start_noise = tf.random.normal(
                    shape=(1, self.img_size, self.img_size, 3)
                )

            else:
                # check if the start_noise has the correct shape
                if start_noise.shape != (1, self.img_size, self.img_size, 3):
                    raise ValueError(
                        f"start_noise should have shape (1, {self.img_size}, {self.img_size}, 3)"
                    )

            # Set the label for the sample(s)
            if poke_type is not None:
                y_label = string_to_onehot(poke_type)
            else:
                index = tf.random.uniform(
                    shape=[], maxval=self.num_classes, dtype=tf.int32
                )
                y_label = tf.one_hot(index, self.num_classes)

            y_label = tf.reshape(y_label, [1, self.num_classes])

            # Generate the sample(s)
            sample = self.sampling_step((start_noise, y_label))
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
