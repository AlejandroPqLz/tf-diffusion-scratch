"""
git_icon.py

Funtionality: This script contains a function to add a GitHub icon and link to the Streamlit app.
"""

# Imports
# =====================================================================
import base64
import streamlit as st
from src_app import APP_FIGURES_DIR

# Define the project directory and image path
GITHUB_ICON_PATH = APP_FIGURES_DIR / "github_icon.png"


# Functions
# =====================================================================
def image_to_base64(image_path: str) -> str:
    """
    Converts an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def add_github_icon():
    """
    Adds a GitHub icon and link to the Streamlit app.
    """
    github_icon_base64 = image_to_base64(GITHUB_ICON_PATH)
    st.markdown(
        f"""
        <style>
        .github-icon-container {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }}
        .github-icon-container a {{
            display: block;
            width: 33px;
            height: 33px;
        }}
        .github-icon-container img {{
            width: 100%;
            height: 100%;
        }}
        </style>
        <div class="github-icon-container">
            <a href="https://github.com/AlejandroPqLz/tf-diffusion-scratch" target="_blank">
                <img src="data:image/png;base64,{github_icon_base64}" alt="GitHub">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
