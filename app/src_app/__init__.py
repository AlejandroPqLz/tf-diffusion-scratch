"""
This file is used to import the necessary data for the Streamlit app.
"""

# Imports
# =====================================================================
from src.utils import PROJECT_DIR

# Generalize paths
APP_FIGURES_DIR = PROJECT_DIR / "figures" / "app_figures"

# Custom css
CUSTOM_CSS = """
    <style>
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
    }
    .css-1aumxhk, .css-1v3fvcr {
        width: 100%;
        max-width: 700px;
        margin: 0 auto;
        text-align: center;
    }
    .css-ffhzg2, .css-13l3l4e {
        display: none;
    }
    .stButton button {
        background-color: #262730;
        color: white;
        width: 100%;
        padding: 1em;
        font-size: 1.2em;
        border-radius: 8px;
    }
    .image-box {
        border: 2px dashed grey;
        padding: 1em;
        margin-bottom: 1em;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        height: 250px;
        max-width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        border-radius: 15px;
    }
    .image-box img {
        max-width: auto;
        max-height: auto;
        object-fit: contain;
    }
    .placeholder-img {
        max-width: 100px;
        max-height: 100px;
        object-fit: contain;
    }
    </style>
"""

# Pokemon types
pokemon_types = [
    "Random",
    "Bug",
    "Dark",
    "Dragon",
    "Electric",
    "Fairy",
    "Fighting",
    "Fire",
    "Flying",
    "Ghost",
    "Grass",
    "Ground",
    "Ice",
    "Normal",
    "Poison",
    "Psychic",
    "Rock",
    "Steel",
    "Water",
]
