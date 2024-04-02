# Diffusion from scratch

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)](https://www.python.org/downloads/release/python-312/)
[![Tensorflow 2.16.1](https://img.shields.io/badge/tensorflow-2.16.1-FF6F00)](https://www.tensorflow.org/)
[![Keras 3.1](https://img.shields.io/badge/keras-3.1-D00000)](https://keras.io/)
[![CUDA 12.3](https://img.shields.io/badge/cuda-12.3-76B900)](https://developer.nvidia.com/cuda-zone)
[![cuDNN 8.9](https://img.shields.io/badge/cudnn-8.9-76B900)](https://developer.nvidia.com/cudnn)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<pre>
 _(`-')     _                                       (`-').->  _                <-. (`-')_      (`-').->             (`-')  (`-')  _ (`-')               (`-').->
( (OO ).-> (_)       <-.        <-.          .->    ( OO)_   (_)         .->      \( OO) )     ( OO)_   _        <-.(OO )  (OO ).-/ ( OO).->  _         (OO )__ 
 \    .'_  ,-(`-')(`-')-----.(`-')-----.,--.(,--.  (_)--\_)  ,-(`-')(`-')----. ,--./ ,--/     (_)--\_)  \-,-----.,------,) / ,---.  /    '._  \-,-----.,--. ,'-'
 '`'-..__) | ( OO)(OO|(_\---'(OO|(_\---'|  | |(`-')/    _ /  | ( OO)( OO).-.  '|   \ |  |     /    _ /   |  .--./|   /`. ' | \ /`.\ |'--...__) |  .--./|  | |  |
 |  |  ' | |  |  ) / |  '--.  / |  '--. |  | |(OO )\_..`--.  |  |  )( _) | |  ||  . '|  |)    \_..`--.  /_) (`-')|  |_.' | '-'|_.' |`--.  .--'/_) (`-')|  `-'  |
 |  |  / :(|  |_/  \_)  .--'  \_)  .--' |  | | |  \.-._)   \(|  |_/  \|  |)|  ||  |\    |     .-._)   \ ||  |OO )|  .   .'(|  .-.  |   |  |   ||  |OO )|  .-.  |
 |  '-'  / |  |'->  `|  |_)    `|  |_)  \  '-'(_ .'\       / |  |'->  '  '-'  '|  | \   |     \       /(_'  '--'\|  |\  \  |  | |  |   |  |  (_'  '--'\|  | |  |
 `------'  `--'      `--'       `--'     `-----'    `-----'  `--'      `-----' `--'  `--'      `-----'    `-----'`--' '--' `--' `--'   `--'     `-----'`--' `--'
</pre>

## :mag: Project Overview

Implementing a **conditioned Denoising Diffsuion Probabilistic Model** (DDPM) on Tensorflow from Scratch for **Pokémon generation** and understanding the theory behind it. In order to achieve it, the Pokemon sprite images dataset from Kaggle will be used: [Pokémon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images) with license: <img src='https://licensebuttons.net/l/zero/1.0/80x15.png'>.

This project has been developed for my **Bachelor's Thesis** in **Data Science and Artificial Intelligence** at Universidad Politécnica de Madrid (UPM).

<div style=\"text-align:center\">
<img src='./figures/readme_figures/poke_red_diffusion_portada.webp'>
</div>

## :open_file_folder: Structure

The **structure** of the repository is as follows:

```tree
📦DiffusionScratch
 ┣ 📂data
 ┃ ┣ 📂interim
 ┃ ┃ ┣ 📜image_paths.json
 ┃ ┃ ┗ 📜pokemon_dict_dataset.json
 ┃ ┣ 📂processed
 ┃ ┃ ┣ 📂pokemon_tf_dataset
 ┃ ┃ ┗ 📜pokedex_cleaned.csv
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂sprites
 ┃ ┃ ┗ 📜pokedex.csv
 ┣ 📂docs
 ┃ ┣ 📂papers
 ┃ ┗ 📂study
 ┣ 📂figures
 ┃ ┣ 📂model_results_figures
 ┃ ┣ 📂notebook_figures
 ┃ ┗ 📂readme_figures
 ┣ 📂models
 ┃ ┗ 📜.gitkeep
 ┣ 📂notebooks
 ┃ ┣ 📜00-Intro-and-Analysis.ipynb
 ┃ ┣ 📜01-Dataset-Creation.ipynb
 ┃ ┣ 📜02-Diffusion-Model-Architecture.ipynb
 ┃ ┣ 📜03-Diffusion-Process.ipynb
 ┃ ┣ 📜04-Training-Diffusion-Model.ipynb
 ┃ ┗ 📜05-Conclusions-and-Results.ipynb
 ┣ 📂src
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📜create_dataset.py
 ┃ ┃ ┣ 📜path_loader.py
 ┃ ┃ ┣ 📜preprocess.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂model
 ┃ ┃ ┣ 📜build_unet.py
 ┃ ┃ ┣ 📜diffusion.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜utils.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂visualization
 ┃ ┃ ┣ 📜visualize.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitattributes
 ┣ 📜.gitignore
 ┣ 📜config.ini
 ┣ 📜config.template.ini
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┗ 📜setup.py
```

## :rocket: Prerequisites

This project contains dependencies outside of the scope of python. Therefore you need to perform additional steps. TODO: add nvidia drivers etc. and add Ubuntu and Windows ways to preprare the enviroment for developing ans using the gpu

- ### Conda Environment
    We will use conda to manage the python environment. You can install it following the [documentation](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
    
    Create the environment:
    
    ```bash
    conda create -n diffusion_env python=3.12.1 -y
    ```
    
    Activate the environment:
    
    ```bash
    conda activate diffusion_env
    ```

- ### External Dependencies
    Once the environment is activated, you can install the [external dependencies](./setup.py) by running the following command:
    
    ```bash
    pip install -e.
    ```

- ### Config.ini
    After installing the external dependencies, take a look to the [config.ini](./config.ini) file in the root of the project. This file will contain all the hyperparameters for the model training.

Now you are ready to go!

## :bar_chart: Data

As mentioned before, the dataset used in this project is the [Pokémon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images) from Kaggle. 

The dataset contains +10,000 Pokémon sprites in PNG format (half of them are shiny variants) in 96x96 resolution from 898 Pokemon in different games, and their corresponding labels that may relate to their design in a CSV file. These aspects will be analyzed deeper in the [00-Intro-and-Analysis.ipynb](./notebooks/00-Intro-and-Analysis.ipynb) notebook.

## :hammer_and_wrench: Usage

After following the steps described in the [Prerequisites](https://github.com/AlejandroPqLz/DiffusionScratch#rocket-prerequisites) section, TODO


## :books: Resources
- Resources and tutorials that have been found useful for this project are located in the [/docs](./docs) folder.
- Git LFS to upload large files into the repository:

    Git Large File Storage (LFS) replaces large files such as datasets, models or weights with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise. 
    For more info, visit: [Git LFS repository](https://github.com/git-lfs/git-lfs/tree/main).
    
    > **WARNING:** Every account using Git Large File Storage receives 1 GiB of free storage and 1 GiB a month of free bandwidth, so in order to avoid any issues uploading heavy files, it is recommended to only upload the heavy files one at a time and do not commit other changes additionally.

## :seedling: Contributing

If you wish to make contributions to this project, please initiate the process by opening an issue or submitting a pull request that encapsulates your proposed modifications.

## :newspaper_roll: License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## :busts_in_silhouette: Contact

Should you have any inquiries or require assistance, please do not hesitate to contact [Alejandro Pequeño Lizcano](pq.lz.alejandro@gmail.com).

Gotta create 'em all!
