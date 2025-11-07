# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    wget \
    curl \
    openssh-client \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /workspace

# Copy common_utils first (dependency)
COPY common_utils /workspace/common_utils

# Copy package files
COPY torchao_float8/pyproject.toml /workspace/torchao_float8/
COPY torchao_float8/uv.lock /workspace/torchao_float8/
COPY torchao_float8/.python-version /workspace/torchao_float8/

# Install dependencies only (leverages cache)
WORKDIR /workspace/torchao_float8
RUN uv python install 3.12
RUN uv venv --python 3.12
RUN uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY torchao_float8 /workspace/torchao_float8

# Install the project itself
RUN uv sync --frozen

# Set the default command
CMD ["/bin/bash", "-c", "echo '[dockerfile] Container Started' && sleep infinity"]