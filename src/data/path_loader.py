"""
path_loader.py

Functionality:
- Load the paths of the images from the dataset
- Save the paths to a file
- Load the paths from a file
"""

# Imports
# =====================================================================
import json
import glob


# class PathLoader
# =====================================================================
class PathLoader:
    """A class to load and save file paths based on a given pattern.

    :param load_pattern: The glob pattern to load the paths from.
    :param save: Whether to save the paths to a file.
    :param save_file: The file to save the paths to.

    """

    def __init__(
        self,
        load_pattern: str,
        save: bool = False,
        save_file: str = "./image_paths.json",
    ):
        self.load_pattern = load_pattern
        self.save = save
        self.save_file = save_file

    def load_paths(self) -> list:
        """Loads file paths based on the glob pattern and optionally saves them.

        :return: A list of file paths.

        """

        print("Loading paths...\n")
        paths = glob.glob(self.load_pattern, recursive=True)
        print(f"- Number of paths loaded: {len(paths)}\n")

        if self.save:

            self.save_paths(paths)

        return paths

    def save_paths(self, paths: list) -> None:
        """Saves the provided paths to a file along with their hashes.

        :param paths: A list of file paths to save.

        """

        try:

            with open(self.save_file, "w", encoding="utf-8") as file:
                json.dump(paths, file, indent=4)

            print(f"Paths loaded and saved to {self.save_file}")

        except IOError as e:

            print(f"Failed to save paths: {e}")


# Aux Functions
# =====================================================================
def get_image_paths(json_path: str) -> list:
    """Loads the image paths from a JSON file.

    :param json_path: The path to the JSON file.
    :return: A list of image paths.

    """

    with open(json_path, "r", encoding="utf-8") as file:
        image_paths = json.load(file)

    return image_paths


# # Test
# # =====================================================================
# SETTINGS_PATH = "../../config.ini"
# config = configparser.ConfigParser()
# config.read(SETTINGS_PATH)
# config = config["paths"]
#
# data_path = config["data_path"]

# loader = PathLoader(
#     load_pattern=f"{data_path}/raw/sprites/**/front/**/*.png",
#     save=True,
#     save_file=f"{data_path}/processed/save_file.json",
# )
# image_paths = loader.load_paths()
