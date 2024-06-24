"""
git_icon.py

Functionality: This script contains a function to add a GitHub icon and link to the Streamlit app.
"""

# Imports
# =====================================================================
import base64
import streamlit as st
from src_app import APP_FIGURES_DIR


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
    github_icon_base64 = image_to_base64(APP_FIGURES_DIR / "github_icon.png")
    st.markdown(
        f"""
        <style>
        .sidebar .github-icon {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
        }}
        .github-icon a {{
            display: block;
            width: 33px;
            height: 33px;
        }}
        .github-icon img {{
            width: 100%;
            height: 100%;
        }}
        </style>
        <div class="github-icon">
            <a href="https://github.com/AlejandroPqLz/tf-diffusion-scratch" target="_blank">
                <img src="data:image/png;base64,{github_icon_base64}" alt="GitHub">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
