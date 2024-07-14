"""
model_loader.py

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

IMG_SIZE = hyperparameters["img_size"]
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
def app_ddpm_model() -> DiffusionModel:
    """
    Load the Denoised Diffusion Probabilistic Model (DDPM) for the Pokémon generator app.

    Returns:
        ddpm_model (DiffusionModel): Loaded DDPM model.
    """
    u_net = build_unet(IMG_SIZE, NUM_CLASSES)
    u_net = build_unet(IMG_SIZE, NUM_CLASSES, dropout_rate=0.1)
    ema_u_net = build_unet(IMG_SIZE, NUM_CLASSES, dropout_rate=0.1)
    ema_u_net.set_weights(u_net.get_weights())
    
    load_path = f"{MODELS_PATH}/final_diffusion_model.weights.h5"

    ddpm_model = DiffusionModel.load_model(
        load_path,
        u_net,
        ema_u_net,
        IMG_SIZE,
        NUM_CLASSES,
        TIMESTEPS,
        BETA_START,
        BETA_END,
        S,
        SCHEDULER,
        ema=0.999
    )

    return ddpm_model