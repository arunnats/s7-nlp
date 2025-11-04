# Dockerfile for a General-Purpose GPU Training Environment

# --- Base Image ---
# Start with an official NVIDIA CUDA image. This image comes with CUDA drivers
# and CuDNN pre-installed, which are necessary for GPU acceleration.
# Using Ubuntu 22.04 to match the host server's environment.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# --- System Dependencies ---
# Set the environment variable to allow package installations without prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install essential tools:
# - python3.10 and python3-pip: For running your Python code.
# - git, curl, wget: For downloading data and code.
# - build-essential: Contains compilers needed for some Python packages.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    nano \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# --- Python Environment ---
# Create a virtual environment to keep Python packages isolated.
RUN python3.10 -m pip install --no-cache-dir virtualenv
RUN python3.10 -m virtualenv /opt/venv

# Activate the virtual environment for all subsequent commands.
ENV PATH="/opt/venv/bin:$PATH"

# --- Install Python Machine Learning Libraries ---
# This is where you install the packages needed for your project.
# Using --no-cache-dir saves space in the final image.
# We are installing PyTorch with CUDA 11.8 support.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the specific libraries for your paper replication project.
# IMPORTANT: Upgrade transformers to version 4.30+ to avoid import errors
RUN pip install --no-cache-dir \
    OpenNMT-py \
    transformers==4.35.2 \
    nltk \
    scikit-learn \
    pandas \
    tqdm \
    datasets \
    sentencepiece \
    rouge-score

# --- Download NLTK Data ---
# Pre-download the NLTK data required for tokenization.
RUN pip install "numpy<2.0"
RUN python -c "import nltk; nltk.download('popular')"

# --- Workspace and Permissions ---
# Set the working directory inside the container. This is where you'll land
# when you start the container.
WORKDIR /workspace

# --- Default Command ---
# When the container starts, it will launch a bash shell, giving you an
# interactive terminal.
ENV TRANSFORMERS_CACHE=/workspace/.cache
ENV HF_HOME=/workspace/.cache
RUN mkdir -p /workspace/.cache && chmod 777 /workspace/.cache
CMD ["zsh"]