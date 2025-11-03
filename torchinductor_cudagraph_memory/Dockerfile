# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Install system dependencies including Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy common_utils first (dependency)
COPY common_utils /workspace/common_utils

# Copy package files
COPY torchinductor_cudagraph_memory/pyproject.toml /workspace/torchinductor_cudagraph_memory/
COPY torchinductor_cudagraph_memory/uv.lock /workspace/torchinductor_cudagraph_memory/

# Install dependencies with uv (leverages cache)
WORKDIR /workspace/torchinductor_cudagraph_memory
RUN uv sync --frozen

# Copy the rest of the application code
COPY torchinductor_cudagraph_memory /workspace/torchinductor_cudagraph_memory

# Set the default command
CMD ["/bin/bash"]

