"""
diffusion_app.py

This script creates a Streamlit app to generate Pokémon using Denoised Diffusion Probabilistic 
Models (DDPM) conditioned on the Pokémon type. The app allows the user to select a Pokémon type
and the number of Pokémon to generate.
"""

# Imports
# =====================================================================
import io
import streamlit as st
from src_app import pokemon_types
from src_app.git_icon import add_github_icon
from src_app.load_diffusion_model import app_ddpm_model

# Streamlit app
# =====================================================================

# App config, title and description
st.set_page_config(layout="wide", page_icon="🎨", page_title="DDPM Pokémon Generator")
add_github_icon()
st.title("DDPM Pokémon Generator 🎨👾")
st.subheader(
    "Generate a Pokémon using Denoised Diffusion Probabilistic Models (DDPM) conditioned on the Pokémon type",
    divider="grey",
)

# Select the Pokémon type, number of Pokémon to generate and size
type_selection = st.selectbox("Select the Pokémon type", pokemon_types)
num_samples = st.number_input(
    "Number of Pokémon to generate",
    1,
    10,
    1,
    help="Select the number of Pokémon to generate, if you select to generate only one Pokemon, you can choose to show the intermediate steps.",
)
size_selection = st.selectbox(
    "Select the size of the Pokémon to generate", ["32x32", "64x64"]
)

if num_samples == 1:
    show_steps = st.checkbox(
        "Show intermediate steps",
        help="If checked, the app will show the intermediate steps of the generation process. Only available when generating one Pokémon.",
    )

# Initialize session state
if "poke_samples" not in st.session_state and "poke_fig" not in st.session_state:
    st.session_state["poke_samples"] = None
    st.session_state["poke_fig"] = None

# Initialize the DDPM model
ddpm_model = app_ddpm_model(int(size_selection.split("x")[0]))

# Generate Pokémon
if st.button("Generate Pokémon"):
    with st.spinner(f"Generating {num_samples} {type_selection} type Pokémon..."):
        st.warning("⚠️ The generation process may take a while.")
        poke_samples = (
            ddpm_model.plot_samples(num_samples)
            if type_selection == "Random"
            else ddpm_model.plot_samples(num_samples, type_selection)
        )

        # Save the figure and the image
        buf = io.BytesIO()
        poke_samples.savefig(buf, format="png")
        buf.seek(0)

        st.session_state["poke_fig"] = poke_samples
        st.session_state["poke_samples"] = buf

        st.success("Pokémon generated successfully! 🎉")

# Display the generated Pokémon and download button (mantaining the state)
if st.session_state["poke_fig"] and st.session_state["poke_samples"]:
    st.pyplot(st.session_state["poke_fig"], use_container_width=False)
    st.download_button(
        label="Download generated Pokémon",
        data=st.session_state["poke_samples"],
        file_name=f"{num_samples}_{type_selection}_type_pokemon_sample.png",
        mime="image/png",
    )
