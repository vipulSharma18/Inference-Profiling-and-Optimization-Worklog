# Docker Setup for Inference Profiling Puzzles

This repository contains problems as separate folders, each with its own Docker container for isolated profiling.

## Building Docker Images

```bash
# Build gemlite_autotune
docker build -f gemlite_autotune/Dockerfile -t ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/gemlite-autotune:main .

# Build torchao_float8
docker build -f torchao_float8/Dockerfile -t ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/torchao-float8:main .

# Build torchinductor_cudagraph_memory
docker build -f torchinductor_cudagraph_memory/Dockerfile -t ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/torchinductor-cudagraph-memory:main .
```

## Running Containers

```bash
# Run gemlite_autotune
docker run --gpus all --cap-add=SYS_ADMIN -d ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/gemlite-autotune:main

# Run torchao_float8
docker run --gpus all --cap-add=SYS_ADMIN -d ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/torchao-float8:main

# Run torchinductor_cudagraph_memory
docker run --gpus all --cap-add=SYS_ADMIN -d ghcr.io/vipulsharma18/inference-profiling-and-optimization-guide/torchinductor-cudagraph-memory:main
```

## CI/CD with GitHub Actions

The repository includes separate GitHub Actions workflows for each problem:

1. `.github/workflows/build-torchao-float8.yml`
2. `.github/workflows/build-gemlite-autotune.yml`
3. `.github/workflows/build-torchinductor-cudagraph.yml`