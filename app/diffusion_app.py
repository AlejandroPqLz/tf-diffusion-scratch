"""
diffusion_app.py

This script creates a Streamlit app to generate Pokémon using Denoised Diffusion Probabilistic 
Models (DDPM) conditioned on the Pokémon type. The app allows the user to select a Pokémon type
and the number of Pokémon to generate.
"""

# Imports
# =====================================================================
import streamlit as st
from src_app.load_diffusion_model import ddpm_model

# Streamlit app
# =====================================================================
st.set_page_config(layout="wide", page_icon="🎨", page_title="DDPM Pokémon Generator")

st.title("DDPM Pokémon Generator")
st.subheader(
    "Generate a Pokémon using Denoised Diffusion Probabilistic Models (DDPM) conditioned on the Pokémon type"
)

pokemon_types = [
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
type_selection = st.selectbox("Select a Pokémon type", pokemon_types)
num_samples = st.number_input("Number of Pokémon to generate", 1, 6, 1)

if st.button("Generate Pokémon"):
    with st.spinner(f"Generating {num_samples} Pokémon of type {type_selection}..."):
        ddpm_model.plot_samples(num_samples, type_selection)
        st.image("pokemon_samples.png")
        st.success("Pokémons generated successfully! 🎉")
