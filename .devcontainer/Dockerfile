# Use the official TensorFlow GPU image as a base image
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables to match the host user and UID
ARG USER_NAME
ARG USER_UID
ENV USER_NAME=$USER_NAME
ENV USER_UID=$USER_UID

# Install Git and set its configuration
RUN apt-get update && apt-get install -y git
# RUN git config --global user.name "Your Name"
# RUN git config --global user.email "youremail@example.com"

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx wget \
 && rm -rf /var/lib/apt/lists/*

# Create a new user in the container with the same UID as the host user
RUN useradd -m -u $USER_UID $USER_NAME

# Set the user as the default user for the container
USER $USER_NAME

# Create a working directory (you can change this to match your needs)
WORKDIR /home/$USER_NAME

# Install Python packages using pip
RUN pip install --upgrade pip && \
    pip install opencv-python tqdm matplotlib scikit-learn scikit-image numba ipykernel pandas np_utils tensorflow-addons tensorflow-datasets tensorflow-hub tensorflow-text keras_nlp --no-cache-dir --no-warn-script-location


# Entry point or command to run when the container starts
CMD bash 
