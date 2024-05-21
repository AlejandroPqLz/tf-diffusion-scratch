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


def load_config(config: configparser.ConfigParser, param_name: str) -> Dict[str, Any]:
    """
    Load the configuration from a file.

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


# Custom parsers
def parse_str(value: str) -> str:
    """
    Parse a string value.

    Args:
        value (str): The string value to parse.

    Returns:
        str: The parsed string value.
    """
    return value.strip('"')


def parse_list_or_tuple(value: str) -> Any:
    """
    Parse a list or tuple value.

    Args:
        value (str): The list or tuple value to parse.

    Returns:
        Any: The parsed list or tuple value.
    """
    return ast.literal_eval(value)
