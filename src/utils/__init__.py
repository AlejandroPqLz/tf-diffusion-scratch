"""
This file is used to import the necessary libraries and load the data.
"""

# Imports
# =====================================================================
from pathlib import Path

# Generalize paths
# =====================================================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data"
CONFIG_PATH = PROJECT_DIR / "config.ini"
