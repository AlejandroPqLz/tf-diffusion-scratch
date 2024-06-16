""""
callbacks.py: Custom Callbacks for the Diffusion Model.

"""

# Imports
# =====================================================================
import configparser
import tensorflow as tf
from src.model.diffusion_funcionality import DiffusionModel
from src.utils.config import parse_config
from src.utils import CONFIG_PATH, MODELS_PATH

# Set config file
# =====================================================================
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
hyperparameters = parse_config(config, "hyperparameters")

# Constants
# =====================================================================
EPOCHS = hyperparameters["epochs"]
BATCH_SIZE = hyperparameters["batch_size"] = 64
IMG_SIZE = hyperparameters["img_size"] = 32
tf.random.set_seed(42)
SAME_NOISE = tf.random.normal(shape=(1, IMG_SIZE, IMG_SIZE, 3))


# Custom Callback for the Diffusion Model
# =====================================================================
class DiffusionCallback(tf.keras.callbacks.Callback):
    """
    Custom Callback for the Diffusion Model that generates samples every N epochs.

    Attributes:
        diffusion_model (DiffusionModel): The diffusion model to generate samples from.
        frequency (int): The frequency at which to generate samples. Defaults to 20.
        poke_type (str): The type of Pokemon to generate samples for. Defaults to None.

    Methods:
        on_epoch_end(epoch, logs): The method that is called at the end of each epoch.
    """

    def __init__(
        self,
        diffusion_model: DiffusionModel,
        frequency: int = 20,
        poke_type: str = None,
    ):
        super(DiffusionCallback, self).__init__()
        self.diffusion_model = diffusion_model
        self.frequency = frequency
        self.poke_type = poke_type

    def on_epoch_end(self, epoch, logs=None):
        """
        The method that is called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): The logs containing the training metrics.
        """
        # Generate samples
        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Generating samples.")
            self.diffusion_model.plot_samples(
                num_samples=1,
                poke_type=self.poke_type,
                start_noise=SAME_NOISE,
                plot_interim=True,
            )

        # Save interim model weights
        if (epoch + 1) % 100 == 0:
            self.diffusion_model.save_weights(
                f"{MODELS_PATH}/interim/diffusion_{IMG_SIZE}x{IMG_SIZE}_batch{BATCH_SIZE}_epochs_{epoch+1}.weights.h5"
            )
