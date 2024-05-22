""""
callbacks.py: Custom Callbacks for the Diffusion Model.

"""

# Imports
# =====================================================================
import tensorflow as tf
from src.model.diffusion_funcionality import DiffusionModel


# TODO: ADD SAVE FUNCTIONALITY TO THE CALLBACKS
# TODO: WHEN PLOTTING SAMPLES, ADD THE OPTION TO SAMPLE THE DIFF PROCESS (FROM NOISE TO IMAGE)
# TODO: ADD LOGS FOR TRAINING METRICS


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
        if (epoch + 1) % self.frequency == 0:
            print(f"Epoch {epoch+1}: Generating samples.")
            self.diffusion_model.plot_samples(num_samples=1, poke_type=self.poke_type)
