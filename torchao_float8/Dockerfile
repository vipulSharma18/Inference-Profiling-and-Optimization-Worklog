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
    less \
    && rm -rf /var/lib/apt/lists/*

# install nsight-systems-cli
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    echo "http://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*

# install gh cli and git-lfs
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    wget -nv -O /tmp/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg && \
    cat /tmp/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Install uv
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Accept build arguments for git repo and commit
ARG GIT_REPO
ARG GIT_COMMIT

# Set working directory
WORKDIR /workspace

# Clone the repository with sparse-checkout
RUN git clone --no-checkout ${GIT_REPO} /workspace && \
    cd /workspace && \
    git sparse-checkout init --cone && \
    git sparse-checkout set torchao_float8 common_utils && \
    git checkout ${GIT_COMMIT}

# Install dependencies
WORKDIR /workspace/torchao_float8
RUN uv python install 3.12 && \
    uv venv --python 3.12 && \
    uv sync --frozen
ENTRYPOINT ["/workspace/torchao_float8/entrypoint.sh"]

# Set the default command
CMD ["/bin/bash", "-c", "echo '[dockerfile] Container Started' && sleep infinity"]