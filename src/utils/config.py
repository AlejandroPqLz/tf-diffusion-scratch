"""
config.py

Functionality: This file contains the code to load the configuration of the project
in a type-safe way.
"""

# Imports
# =====================================================================
import configparser
from typing import Dict, Any
import ast


def parse_config(config: configparser.ConfigParser, param_name: str) -> Dict[str, Any]:
    """
    Parse the configuration of the project in a type-safe way.

    Args:
        config (configparser.ConfigParser): The configuration object.
        param_name (str): The name of the section in the configuration file.

    Returns:
        Dict[str, Any]: The configuration of the project.
    """

    if param_name not in config:
        raise ValueError(f"Section {param_name} not found in the configuration file.")

    section = config[param_name]
    parsed_config = {}

    for key, value in section.items():
        value_str, value_type = value.rsplit(",", 1)
        value_str = value_str.strip()
        value_type = value_type.strip()

        try:
            if value_type == "str":
                # Special handling for string values
                parsed_value = value_str
            else:
                # Evaluate the value using the type as a function
                parsed_value = ast.literal_eval(value_str)
            parsed_config[key] = parsed_value
        except Exception as e:
            raise ValueError(
                f"Error parsing {key} with value {value_str} as {value_type}: {e}"
            ) from e

    return parsed_config
