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
from src_app import APP_FIGURES_DIR, CUSTOM_CSS, pokemon_types
from src_app.icon_loader import image_to_base64, add_github_icon
from src_app.model_loader import app_ddpm_model

# Load images
# =====================================================================
pokeball_img = image_to_base64(APP_FIGURES_DIR / "pokeball.png")
ultraball_img = image_to_base64(APP_FIGURES_DIR / "ultraball.png")
exp_share_img = image_to_base64(APP_FIGURES_DIR / "exp_share.png")
placeholder_img = image_to_base64(APP_FIGURES_DIR / "placeholder.png")

# Streamlit app configuration
# =====================================================================
st.set_page_config(
    layout="wide",
    page_icon=f"data:image/png;base64,{ultraball_img}",
    page_title="DDPM Pokémon Generator",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)  # Add custom CSS

# Sidebar
# =====================================================================
with st.sidebar:
    SIDEBAR_MSG = """
        <h1>
        Generation settings
        </h1>
        <hr style="border:1px solid grey; margin-top: 1px;">
    """
    st.markdown(SIDEBAR_MSG, unsafe_allow_html=True)

    type_selection = st.selectbox("Pokémon type", pokemon_types)
    num_samples = st.number_input(
        "Number of Pokémon to generate",
        1,
        10,
        1,
        help="Select the number of Pokémon to generate, if you select to generate only one Pokémon, you can choose to show the intermediate steps.",
    )
    size_selection = st.selectbox("Size of the Pokémon to generate", ["32x32", "64x64"])

    show_steps = False
    if num_samples == 1:
        show_steps = st.checkbox(
            "Show intermediate steps",
            help="If checked, the app will show the intermediate steps of the generation process. Only available when generating one Pokémon.",
        )

    add_github_icon()  # Add GitHub icon at the bottom-left of the sidebar

# Initialize session state
# =====================================================================
if "poke_samples" not in st.session_state and "poke_fig" not in st.session_state:
    st.session_state["poke_samples"] = None
    st.session_state["poke_fig"] = None

# Title and subtitle
# =====================================================================
st.markdown(
    f"""
    <h1>
        DDPM Pokémon Generator
        <img src="data:image/png;base64,{pokeball_img}" alt="pokeball" width="42" height="42">
        <img src="data:image/png;base64,{exp_share_img}" alt="exp_share" width="42" height="42">
    </h1>
    """,
    unsafe_allow_html=True,
)

st.subheader(
    "Generate new Pokémon using Denoised Diffusion Probabilistic Models (DDPM) conditioned on the Pokémon type",
    divider="grey",
)

# Display placeholder or generated Pokémon
# =====================================================================
image_box = st.empty()
with image_box:
    if st.session_state["poke_samples"]:
        st.image(st.session_state["poke_samples"], use_column_width=False)
    else:
        st.markdown(
            f"""
            <div class="image-box">
                <img class="placeholder-img" src="data:image/png;base64,{placeholder_img}" alt="placeholder" width="69" height="69">
            </div>
            """,
            unsafe_allow_html=True,
        )

# Generate Pokémon
# =====================================================================
if st.button("Generate Pokémon"):
    try:
        ddpm_model = app_ddpm_model(int(size_selection.split("x")[0]))

        with image_box.container():
            st.markdown(
                f"""
                <div class="image-box">
                    <div role="status">
                        <span>Generating {num_samples} {type_selection} type Pokémon...</span>
                    </div>
                </div>
                <br>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "This process takes 1-2 minutes per sample, please wait...",
                icon="⚠️",
            )

        if num_samples == 1 and show_steps:
            poke_samples = (
                ddpm_model.plot_samples(num_samples, plot_interim=True)
                if type_selection == "Random"
                else ddpm_model.plot_samples(
                    num_samples, type_selection, plot_interim=True
                )
            )
        else:
            poke_samples = (
                ddpm_model.plot_samples(num_samples)
                if type_selection == "Random"
                else ddpm_model.plot_samples(num_samples, type_selection)
            )

        buf = io.BytesIO()
        poke_samples.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        st.session_state["poke_fig"] = poke_samples
        st.session_state["poke_samples"] = buf

        with image_box:
            if show_steps or num_samples >= 7:
                st.image(buf, use_column_width=True)
            else:
                st.image(buf, use_column_width=False)

        st.success(
            f"Successfully generated {num_samples} {type_selection} type new Pokémon! 🎉"
        )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please refresh the page and try again.")

# Display the download button
# =====================================================================
if st.session_state["poke_samples"]:
    st.download_button(
        label="Download generated Pokémon",
        data=st.session_state["poke_samples"],
        file_name=(
            f"{num_samples}_{type_selection}_type_pokemon_sample.png"
            if not show_steps
            else f"{num_samples}_{type_selection}_type_pokemon_sample_interim.png"
        ),
        mime="image/png",
    )
