# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Install system dependencies including Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    wget \
    build-essential \
    vim \
    ca-certificates \
    openssh-server \
    openssh-client \
    procps \
    && rm -rf /var/lib/apt/lists/*

# add keyring for gh/github cli for git creds management
RUN mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& mkdir -p -m 755 /etc/apt/sources.list.d \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& apt-get update \ 
&& apt-get install -y --no-install-recommends gh \
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
COPY gemlite_autotune/pyproject.toml /workspace/gemlite_autotune/
COPY gemlite_autotune/uv.lock /workspace/gemlite_autotune/

# Install dependencies only (leverages cache)
WORKDIR /workspace/gemlite_autotune
RUN uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY gemlite_autotune /workspace/gemlite_autotune

# Install the project itself
RUN uv sync --frozen

# Set the default command
CMD ["/bin/bash"]

