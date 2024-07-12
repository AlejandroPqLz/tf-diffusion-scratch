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
        ema_model (tf.keras.Model): The model used to store the EMA weights.
        img_size (int): The size of the input images.
        num_classes (int): The number of classes in the dataset.
        timesteps (int): The total number of diffusion steps.
        beta_start (float): The starting value of beta (noise level).
        beta_end (float): The ending value of beta (noise level).
        s (float): The scale factor for the variance curve in the 'cosine' scheduler.
        scheduler (str): The type of noise schedule ('cosine' or 'linear').
        ema (float): The exponential moving average factor.

    Methods:
        compile(loss, optimizer, **kwargs): Compile the model with the specified loss and optimizer.
        train_step(data): The training step for the diffusion model.
        sampling_step(data): The sampling step for the diffusion model.
        forward_diffusion(x_0, t): Diffuse the data by adding noise to the input image.
        beta_scheduler(): Generates a schedule for beta values according to the specified type.
        _gather(tensor, index): Extract the value at the specified index from the tensor.
        plot_samples(num_samples, poke_type): Generate and plot samples from the diffusion model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        ema_model: tf.keras.Model,
        img_size: int,
        num_classes: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        s: float,
        scheduler: str,
        ema: float = 0.999,
    ):

        super().__init__()
        self.model = model
        self.ema_model = ema_model
        self.img_size = img_size
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.scheduler = scheduler
        self.ema = ema

        self.beta = tf.constant(self.beta_scheduler(), tf.float32)
        self.alpha = tf.constant(1 - self.beta, tf.float32)
        self.alpha_cumprod = tf.constant(tf.math.cumprod(self.alpha), tf.float32)
        self.sigma = tf.constant(tf.sqrt(self.beta), tf.float32)  # σ_t = sqrt(β_t)

        self.compiled_loss = tf.keras.losses.MeanSquaredError()
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
        self.compiled_loss = loss
        self.optimizer = optimizer

    def _gather(self, tensor: tf.Tensor, index: int) -> tf.Tensor:
        """
        Extract the value at the specified index from the tensor,
        then reshape to [batch_size, 1, 1, 1] for broadcasting.

        Args:
            tensor (tf.Tensor): The tensor to gather values from.
            index (int): The index to gather values for.

        Returns:
            tf.Tensor: The gathered tensor values reshaped for broadcasting.
        """
        tensor_t = tf.gather(tensor, index)
        return tf.reshape(tensor_t, [-1, 1, 1, 1])

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
        alpha_cumprod_t = self._gather(self.alpha_cumprod, t)

        mean = tf.sqrt(alpha_cumprod_t) * x_0
        variance = 1 - alpha_cumprod_t

        x_t = mean + tf.sqrt(variance) * noise

        return x_t, noise

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

        # Randomly mask out input_label
        mask = tf.random.uniform(
            shape=(batch_size, self.num_classes), minval=0, maxval=1
        )
        mask = tf.cast(mask > 0.1, tf.float32)  # 10% of the labels are masked
        input_label = tf.cast(input_label, tf.float32)
        y = tf.cast(input_label * mask, tf.int32)

        # 1: repeat ------

        # 3: t ~ U(0, T): Generate a random timestep for each image in the batch
        t = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.timesteps, dtype=tf.int32
        )

        # 2: x_0 ~ q(x_0), 4: ε_t ~ N(0, I)
        x_t, target_noise = self.forward_diffusion(input_data, t)

        # 5: Take a gradient descent step on
        with tf.GradientTape() as tape:
            predicted_noise = self.model([x_t, y, t], training=True)
            loss = self.compiled_loss(target_noise, predicted_noise)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.model.weights, self.ema_model.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 6: until convergence ------

        # Update the metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(target_noise, predicted_noise)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: tuple) -> dict:
        """
        The validation step for the diffusion model.

        Args:
            data (tuple): A tuple containing the input data and labels.

        Returns:
            dict: A dictionary containing the updated metrics.
        """
        # Unpack the data
        input_data, input_label = data
        batch_size = tf.shape(input_data)[0]

        # Randomly mask out input_label
        mask = tf.random.uniform(
            shape=(batch_size, self.num_classes), minval=0, maxval=1
        )
        mask = tf.cast(mask < 0.1, tf.float32)  # 10% of the labels are masked
        input_label = tf.cast(input_label, tf.float32)
        y = tf.cast(input_label * mask, tf.int32)

        # Generate a random timestep for each image in the batch
        t = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.timesteps, dtype=tf.int32
        )

        # Forward diffusion to get the noisy image and target noise
        x_t, target_noise = self.forward_diffusion(input_data, t)

        # Predict the noise
        predicted_noise = self.model([x_t, y, t], training=False)
        loss = self.compiled_loss(target_noise, predicted_noise)

        # Update the metrics
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
        x_T = x_t
        shape_x_t = tf.shape(x_t)

        # 2: for t = T, . . . , 1 do: Reverse the diffusion process
        time.sleep(0.4)
        interim = dict()
        inv_process = reversed(range(1, self.timesteps))
        for t_ in tqdm(inv_process, "Sampling sprite...", total=self.timesteps - 1):
            t = tf.cast(tf.fill(shape_x_t[0], t_), tf.int32)  # shape = (batch_size,)

            # 3: z ~ N(0, I) if t > 1, else z = 0:
            # Sample noise, except for last image (t=1, x_t-1 = x_0)
            z = tf.random.normal(shape=shape_x_t) if t_ > 1 else tf.zeros_like(x_t)

            # 4: x_{t-1} = (x_t - (1 - α_t) / sqrt(1 - α_cumprod_t) * ε_θ) / sqrt(α_t) + σ_t * z
            predicted_noise = self.ema_model([x_t, y_t, t], training=False)
            alpha_t = self._gather(self.alpha, t)
            alpha_cumprod_t = self._gather(self.alpha_cumprod, t)
            sigma_t = self._gather(self.sigma, t)

            x_t = (
                x_t - (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t) * predicted_noise
            ) / tf.sqrt(alpha_t) + sigma_t * z

            # Save the intermediate steps for later plotting
            if t_ == self.timesteps - 1:
                interim[t_] = x_T
            if t_ % 100 == 0 or t_ == 1:
                interim[t_ - 1] = x_t  # x_t = x_{t-1}

        # 5: end for
        return x_t, interim  # 6: return x_0

    def plot_samples(
        self,
        num_samples: int = 1,
        poke_type: str = None,
        start_noise: tf.Tensor = None,
        plot_interim: bool = False,
        app: bool = False,
    ) -> plt.Figure:
        """
        Generate and plot samples from the diffusion model.

        Args:
            num_samples (int): The number of samples to generate and plot.
            poke_type (str): The type of Pokemon to generate samples for.
            If None, a random type is chosen.
            start_noise (tf.Tensor): The starting noise tensor. If None, random noise is used.
            plot_interim (bool): Whether to plot the intermediate steps of the diffusion process.
            app (bool): Whether to return the figure for the Streamlit app.

        Returns:
            plt.Figure: The figure containing the generated samples.
        """
        if not plot_interim:
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
            sample, interim = self.sampling_step((start_noise, y_label))

            # Plot the samples or the interim steps
            # =====================================================================
            if not plot_interim:
                sample = tf.squeeze(sample)  # remove the batch dimension
                # Scale to [0, 1] for plotting
                sample = (sample - tf.reduce_min(sample)) / (
                    tf.reduce_max(sample) - tf.reduce_min(sample)
                )

                # Plot the samples
                axs[i].imshow(sample)
                axs[i].title.set_text(onehot_to_string(y_label))
                axs[i].axis("off")

            else:

                for t, step in interim.items():
                    interim[t] = tf.squeeze(step)

                for t, step in interim.items():
                    interim[t] = (step - tf.reduce_min(step)) / (
                        tf.reduce_max(step) - tf.reduce_min(step)
                    )

                # Plot the interim steps
                _, axs = plt.subplots(1, len(interim), figsize=(len(interim) * 2, 3))
                if len(interim) == 1:
                    axs = [axs]

                for j, (t, step) in enumerate(interim.items()):
                    axs[j].imshow(step)
                    axs[j].title.set_text(f"{onehot_to_string(y_label)}: t={t}")
                    axs[j].axis("off")

        return plt.gcf() if app else plt.show()

    @staticmethod
    def load_model(
        model_path: str,
        base_model: tf.keras.Model,
        ema_model: tf.keras.Model,
        img_size: int,
        num_classes: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        s: float,
        scheduler: str,
        ema: float = 0.999,
    ) -> tf.keras.Model:
        """
        Load a trained model from a file.

        Args:
            model_path (str): The path to the model file.
            base_model (tf.keras.Model): The base model to which the diffusion process is added.
            ema_model (tf.keras.Model): The model used to store the EMA weights.
            img_size (int): The size of the input images.
            num_classes (int): The number of classes in the dataset.
            timesteps (int): The total number of diffusion steps.
            beta_start (float): The starting value of beta (noise level).
            beta_end (float): The ending value of beta (noise level).
            s (float): The scale factor for the variance curve in the 'cosine' scheduler.
            scheduler (str): The type of noise schedule ('cosine' or 'linear').
            ema (float): The exponential moving average factor.

        Returns:
            model: The trained model with the diffusion functionality.
        """
        model = DiffusionModel(
            model=base_model,
            ema_model=ema_model,
            img_size=img_size,
            num_classes=num_classes,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            s=s,
            scheduler=scheduler,
            ema=ema,
        )

        model.load_weights(model_path)

        return model
