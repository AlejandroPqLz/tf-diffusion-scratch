"""
path_loader.py

Functionality:
- Load the paths of the images from the dataset
- Save the paths to a file
- Load the paths from a file
"""

# Imports and setup
# =====================================================================
import json
import logging
import glob
from pathlib import Path
from typing import List
from src.utils.utils import DATA_PATH

logging.basicConfig(level=logging.INFO)


# class PathLoader
# =====================================================================
class PathLoader:
    """A class to load and save file paths based on a given pattern.

    Attributes:
        load_pattern (str): The glob pattern to load the file paths.
        save_file (str): The file to save the file paths. Default to "./image_paths.json".
    """

    def __init__(
        self,
        load_pattern: str,
        save_file: str = "./image_paths.json",
    ):
        self.load_pattern = load_pattern
        self.save_file = Path(save_file)

    def load_paths(self, save: bool = False) -> List[str]:
        """Loads file paths based on the glob pattern and optionally saves them.

        Args:
            save (bool): Whether to save the file paths. Defaults to False.

        Returns:
            List[str]: A list of file paths.
        """
        if self.save_file.exists():
            saved_paths = PathLoader.load_paths_from_file(self.save_file)
            first_path = Path(saved_paths[0])
            pattern_name = Path(self.load_pattern).name

            if saved_paths and first_path.match(pattern_name):
                logging.info("Using existing paths from %s", self.save_file)
                return saved_paths

            else:
                logging.info(
                    "Existing paths in %s do not match the current pattern. Loading new paths.",
                    self.save_file,
                )

        logging.info("Loading new paths from %s", self.load_pattern)
        paths = glob.glob(self.load_pattern, recursive=True)

        if save:
            self.save_paths(paths)

        return paths

    def save_paths(self, paths: List[str]) -> None:
        """Saves the provided paths to a file.

        Args:
            paths (List[str]): A list of file paths.
        """
        try:
            with self.save_file.open("w", encoding="utf-8") as file:
                json.dump(paths, file, indent=4)

            logging.info("Paths loaded and saved to %s", self.save_file)

        except IOError as e:
            logging.error("Failed to save paths to %s: %s", self.save_file, e)

    @staticmethod
    def load_paths_from_file(save_file: str) -> List[str]:
        """Loads the paths from a file.

        Args:
            save_file (str): The path to the JSON file.

        Returns:
            List[str]: A list of paths loaded from the file.
        """
        try:
            with save_file.open("r", encoding="utf-8") as file:
                paths = json.load(file)
                logging.info("%d paths loaded from %s", len(paths), save_file)
                return paths

        except FileNotFoundError as e:
            logging.error("Failed to load paths from %s: %s", save_file, e)
            return []

        except json.JSONDecodeError as e:
            logging.error("Invalid JSON format in %s: %s", save_file, e)
            return []


# Main
# =====================================================================
def main() -> None:
    """Main function to load the image paths from the dataset.

    Args:
        config_path (str): The path to the configuration file.
    """

    loader = PathLoader(
        load_pattern=f"{DATA_PATH}/raw/sprites/**/front/**/*.png",
        save_file=f"{DATA_PATH}/processed/save_file.json",
    )
    image_paths = loader.load_paths(save=True)
    logging.info("Loaded %d image paths to %s", len(image_paths), loader.save_file)


if __name__ == "__main__":
    main()
