"""
load_diffusion_model.py

Functionality: Load the Denoised Diffusion Probabilistic Model (DDPM) for the Pokémon generator app.
"""

# Imports
# =====================================================================
import configparser
import tensorflow as tf
from src.model.build_model import build_unet
from src.utils import CONFIG_PATH, MODELS_PATH
from src.utils.config import parse_config
from src.model.diffusion_funcionality import DiffusionModel

# Use the GPU
# =====================================================================
gpus_list = tf.config.list_physical_devices("GPU")
gpu = gpus_list[0]
tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs Available: ", gpus_list)

# Set config file
# =====================================================================
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

hyperparameters = parse_config(config, "hyperparameters")

NUM_CLASSES = hyperparameters["num_classes"]
BATCH_SIZE = hyperparameters["batch_size"]
EPOCHS = hyperparameters["epochs"]

TIMESTEPS = hyperparameters["timesteps"]
SCHEDULER = hyperparameters["scheduler"]
BETA_START = hyperparameters["beta_start"]
BETA_END = hyperparameters["beta_end"]
S = hyperparameters["s"]


# Get the loaded model
# =====================================================================
def app_ddpm_model(img_size: int) -> DiffusionModel:
    """
    Load the Denoised Diffusion Probabilistic Model (DDPM) for the Pokémon generator app.

    Args:
        size (int): Image size (32 or 64).

    Returns:
        ddpm_model (DiffusionModel): Loaded DDPM model.
    """
    u_net = build_unet(img_size, NUM_CLASSES)
    save_path = f"{MODELS_PATH}/diffusion_{img_size}x{img_size}_batch{BATCH_SIZE}_epochs{EPOCHS}.weights.h5"

    ddpm_model = DiffusionModel.load_model(
        save_path,
        u_net,
        img_size,
        NUM_CLASSES,
        TIMESTEPS,
        BETA_START,
        BETA_END,
        S,
        SCHEDULER,
    )

    return ddpm_model
