"""
setup.py

Funcionality: This file is used to create a package that can be installed using pip. 
It is used to install the package in the local environment.

"""

import setuptools

requirements = [
    "numpy >=1.26.4",
    "pandas >= 2.2.1",
    "matplotlib >= 3.8.3",
    "tqdm >= 4.66.2",
    # "tensorflow[and-cuda] >= 2.16",
    "tensorflow >= 2.16",
    "keras >= 3.1",
    "scikit-learn >= 1.4.1.post1",
    "scipy >= 1.12.0",
    "mlflow",
    "pexpect",
    "ipykernel",
]

setuptools.setup(
    name="poke_diffusion",
    version="0.0.1",
    author="Alejandro pequeño Lizcano",
    author_email="pq.lz.alejandro@gmail.com",
    description="Package for implementing a DDPM on Tensorflow from Scratch",
    long_description_content_type="text/markdown",
    url="https://github.com/AlejandroPqLz/DiffusionScratch",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
