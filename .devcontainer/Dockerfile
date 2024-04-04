# Use the official TensorFlow GPU image as a base image
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables to match the host user and UID
ARG USER_NAME
ARG USER_UID
ENV USER_NAME=${USER_NAME}
ENV USER_UID=${USER_UID}

# Install Git and system dependencies for packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Uncomment to set global Git configuration if needed
# RUN git config --global user.name "Your Name" && \
#     git config --global user.email "youremail@example.com"

# Create a new user in the container with the same UID as the host user to avoid permission issues
RUN useradd -m -u ${USER_UID} ${USER_NAME}

# Set the user as the default user for the container
USER ${USER_NAME}

# Set the working directory to the user's home directory (change this to match your needs)
WORKDIR /home/${USER_NAME}

# Install Python packages using pip
RUN pip install --upgrade pip && \
    pip install gradio ipykernel keras==3.1 matplotlib mlflow numba numpy pandas pexpect \
    tqdm --no-cache-dir --no-warn-script-location

# Set the default command or entry point for the container
CMD ["bash"]