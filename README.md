# Diffusion from scratch
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB)](https://www.python.org/downloads/release/python-312/)
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

Implementing a **conditioned Denoising Diffsuion Probabilistic Model** (DDPM) on Tensorflow from Scratch for **Pokémon generation** and understanding the mathematics and theory behind it. Therefore to achive this goal, the Pokémon sprites dataset will be used: [Pokémon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images) with license: <img src='https://licensebuttons.net/l/zero/1.0/80x15.png'>.

This project has been developed for my **Bachelor's Thesis** in **Data Science and Artificial Intelligence** at Universidad Politécnica de Madrid (UPM).

> <span style="color: red; font-size: 1.5em;">&#9888;</span> **NOTE:** Since this project is for a spanish college bachelor's thesis, the documentation markdowns in the notebooks are in spanish. However, the code and comments are in english.

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
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┗ 📜setup.py
```

## :rocket: Prerequisites

This project contains dependencies outside of the scope of python. Therefore you need to perform additional steps.

### 1. OS Requirements
---

It is recommended to use a **Linux** distribution for this project, since it is the most common OS for data science and machine learning tasks and for that reason, NVIDIA GPU configurations are easier to set up. However, you can also use **Windows** with **WSL2** or **MacOS**. The requirements for each OS are as follows:

<table>
    <thead>
        <tr>
            <th>Windows</th>
            <th>Linux (Ubuntu)</span></a></th>
            <th>MacOS</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <ul>
                    <li>Windows 11</li>
                    <li>NVIDIA GPU with CUDA support</li>
                    <li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">Download and set up WSL2.</a>
                    <li>Install Ubuntu from the Microsoft Store</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Ubuntu 22.04 or later</li>
                    <li>NVIDIA GPU with CUDA support</li>
                    <li><a href="https://docs.docker.com/compose/install/">Install Docker Compose on Ubuntu</a></li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>macOS 12.0 or later (Get the latest beta)</li>
                    <li>Mac computer with Apple silicon or AMD GPUs</li>
                    <li>Xcode command-line tools: <code>xcode-select — install</code></li>    
                </ul>
            </td>
        </tr>
    </tbody>
</table>

### 2. NVIDIA GPU Configuration (Windows and Linux)
---

In order to use the GPU for training the model, you need to install the NVIDIA drivers, CUDA and cuDNN. Eventhough the project is developed in Tensorflow and therefore not all CUDA and cuDNN versions are compatible with the version of Tensorflow used, for the GPU to work properly, the versions of CUDA and cuDNN and the NVIDIA drivers must be the most recent ones.

#### 2.1 Install NVIDIA drivers:

<table>
    <thead>
        <tr>
            <th>Windows</th>
            <th>Linux (Ubuntu)</span></a></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <ul>
                    <li>Download the latest NVIDIA drivers </br> for your GPU on Windowns from the <a href="https://www.nvidia.com/download/index.aspx?lang=en-us">NVIDIA website</a></li>
                    <li>Install the <code>.exe</code> file</li> and follow the instructions
                    <li>Chech the driver installation: </br>
                    <code>nvidia-smi</code></li>
                </ul>
            </td>
            <td>
                <ul>
                    <li> Update and upgrade the system: </br>
                    <code>sudo apt update && sudo apt upgrade</code></li>
                    <li> Remove previous NVIDIA installations: </br>
                    <code>sudo apt autoremove nvidia* --purge</code></li>
                    <li> Check Ubuntu devices: </br>
                    <code>ubuntu-drivers devices</code></li>
                    <li> Install the recommended NVIDIA driver (its version is tagged with recommended): </br>
                    <code>sudo apt-get install nvidia-driver-&ltdriver_number&gt</code></li>
                    <li> Reboot the system: </br>
                    <code>reboot</code></li>
                    <li>Chech the driver installation: </br>
                    <code>nvidia-smi</code></li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

After these steps, when executing the `nvidia-smi` command, you should see the following output:

```bash
user@user:~$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.39.01    Driver Version: 510.39.01    CUDA Version: 12.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8    10W /  N/A |      0MiB /  5944MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
#### 2.2 Install CUDA toolkit:

**- Windows:** [Install CUDA toolkit on Windows](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) </br>
**- WSL2:** [Install CUDA toolkit on WSL2](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) </br>
**- Ubuntu:** [Install CUDA toolkit on Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

#### 2.3 Install cuDNN:

**- WSL2:** [Install cuDNN](https://developer.nvidia.com/cudnn) </br>
**- Ubuntu:** [Install cuDNN](https://developer.nvidia.com/cudnn)

### 3. Windows Subsystem for Linux (WSL2) Set up
---

After installing the NVIDIA drivers, CUDA and cuDNN, you need to set up WSL2 to use the GPU for training the model. To do this, follow the steps below:

#### 3.1  Conda Environment

We will use conda to manage the python environment. You can install it following the [Miniconda instalation guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). After installing miniconda, create a new environment with the following command:
    
```bash
    # Create the environment
    conda create -n diffusion_env python=3.12 -y
    
    # Activate the environment
    conda activate diffusion_env
```

#### 3.2  CUDA and cuDNN compatible versions

Since the model is implemented in Tensorflow, you need to install the versions of CUDA and cuDNN that are compatible with the version of Tensorflow you are using. For more information, visit the [Tensorflow versions compatibility](https://www.tensorflow.org/install/source?hl=es#gpu). For this project, since we are using Tensorflow 2.16.1, we need to install CUDA 12.3 and cuDNN 8.9, todo do so, just execute the following commands:

```bash
    # Install CUDA 12.3
    conda install nvidia/label/cuda-12.3.2::cuda-toolkit
    
    # Install cuDNN 8.9
    conda install -c conda-forge cudnn=8.9
```

And make the following changes in the environment variables for using CUDA and cuDNN after activating the environment:

```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
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
- Conda environment installation and management: [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
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
